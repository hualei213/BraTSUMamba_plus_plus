import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.MDAF import MDAF
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class MultiPlaneWaveletFeatureExtractor(nn.Module):
    """多平面小波特征提取器（使用3D小波）"""

    def __init__(self, wavelet_level: int = 1):
        super(MultiPlaneWaveletFeatureExtractor, self).__init__()
        self.wavelet_level = wavelet_level

        # 使用2D小波在三个平面
        self.haar_axial = HaarWavelet2D(level=wavelet_level)
        self.haar_coronal = HaarWavelet2D(level=wavelet_level)
        self.haar_sagittal = HaarWavelet2D(level=wavelet_level)

    def extract_2d_features(self, x: torch.Tensor, plane: str):
        """2D平面特征提取"""
        B, C, D, H, W = x.shape

        if plane == 'axial':
            # 轴状面: (B, C, D, H, W) -> (B*D, C, H, W)
            x_reshaped = x.permute(0, 1, 3, 4, 2).contiguous()
            x_reshaped = x_reshaped.reshape(B * D, C, H, W)
            low, high = self.haar_axial(x_reshaped)
            low = low.view(B, C, H, W, D).permute(0, 1, 4, 2, 3).contiguous()
            high = high.view(B, C, H, W, D).permute(0, 1, 4, 2, 3).contiguous()

        elif plane == 'coronal':
            # 冠状面: (B, C, D, H, W) -> (B*H, C, W, D)
            x_reshaped = x.permute(0, 1, 4, 2, 3).contiguous()
            x_reshaped = x_reshaped.reshape(B * H, C, W, D)
            low, high = self.haar_coronal(x_reshaped)
            low = low.view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous()
            high = high.view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous()

        elif plane == 'sagittal':
            # 矢状面: (B, C, D, H, W) -> (B*W, C, H, D)
            x_reshaped = x.permute(0, 1, 3, 2, 4).contiguous()
            x_reshaped = x_reshaped.reshape(B * W, C, H, D)
            low, high = self.haar_sagittal(x_reshaped)
            low = low.view(B, C, H, D, W).permute(0, 1, 3, 2, 4).contiguous()
            high = high.view(B, C, H, D, W).permute(0, 1, 3, 2, 4).contiguous()

        return low, high

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量 [B, C, D, H, W]

        Returns:
            low_freq: 低频特征 [B, C, D, H, W]
            high_freq: 高频特征 [B, C, D, H, W]
        """
        # 使用2D多平面小波（向后兼容）
        low_axial, high_axial = self.extract_2d_features(x, 'axial')
        low_coronal, high_coronal = self.extract_2d_features(x, 'coronal')
        low_sagittal, high_sagittal = self.extract_2d_features(x, 'sagittal')

        #融合
        low_freq = (low_axial + low_coronal + low_sagittal)
        high_freq = (high_axial + high_coronal + high_sagittal)

        return low_freq, high_freq


class HaarWavelet2D(nn.Module):

    def __init__(self, level: int = 1):
        super(HaarWavelet2D, self).__init__()
        self.level = level

        self.register_buffer('ll_kernel', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4)
        self.register_buffer('lh_kernel', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32).view(1, 1, 2, 2) / 4)
        self.register_buffer('hl_kernel', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32).view(1, 1, 2, 2) / 4)
        self.register_buffer('hh_kernel', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        ll_kernel = self.ll_kernel.repeat(C, 1, 1, 1)
        lh_kernel = self.lh_kernel.repeat(C, 1, 1, 1)
        hl_kernel = self.hl_kernel.repeat(C, 1, 1, 1)
        hh_kernel = self.hh_kernel.repeat(C, 1, 1, 1)

        low_freq = x.clone()
        high_freq = torch.zeros_like(x)

        for l in range(self.level):
            stride = 2 ** l
            if H // stride < 2 or W // stride < 2:
                break

            ll = F.conv2d(low_freq, ll_kernel, stride=stride, padding=0, groups=C)
            lh = F.conv2d(low_freq, lh_kernel, stride=stride, padding=0, groups=C)
            hl = F.conv2d(low_freq, hl_kernel, stride=stride, padding=0, groups=C)
            hh = F.conv2d(low_freq, hh_kernel, stride=stride, padding=0, groups=C)


            current_high = torch.abs(lh) + torch.abs(hl) + torch.abs(hh)


            current_high = F.interpolate(current_high, size=(H, W),
                                         mode='bilinear', align_corners=False)


            high_freq += current_high

            # 更新低频
            low_freq = F.interpolate(ll, size=(H, W),
                                     mode='bilinear', align_corners=False)

        # 归一化
        if self.level > 0:
            high_freq = high_freq / self.level

        return low_freq, high_freq


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 use_frequency_decomposition: bool = True,
                 use_mdaf_fusion: bool = True,
                 frequency_levels: int = 1
                 ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # 标准解码组件
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))

            stages.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.use_frequency_decomposition = use_frequency_decomposition

        # 小波分解模块 - 为每个解码阶段创建一个
        if self.use_frequency_decomposition:
            self.wavelet_extractors = nn.ModuleList([
                MultiPlaneWaveletFeatureExtractor(
                    wavelet_level=frequency_levels
                ) for _ in range(n_stages_encoder - 1)
            ])
        else:
            self.wavelet_extractors = None

        self.use_mdaf_fusion = use_mdaf_fusion

        # MDAF融合模块
        if self.use_mdaf_fusion:
            self.mdaf_modules = nn.ModuleList()
            for s in range(1, n_stages_encoder):
                skip_channels = encoder.output_channels[-(s + 1)]
                mdaf = MDAF(
                    dim=skip_channels,
                    num_heads=8,
                    LayerNorm_type='WithBias'
                )
                self.mdaf_modules.append(mdaf)
        else:
            self.mdaf_modules = None

    def forward(self, skips):
        """前向传播"""
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            # 上采样
            x = self.transpconvs[s](lres_input)

            # 获取跳跃连接特征
            skip_feature = skips[-(s + 2)]

            # 拼接跳跃连接和上采样特征
            x_stage = torch.cat((x, skip_feature), 1)

            # 通过卷积块
            x_stage = self.stages[s](x_stage)

            # 小波频率分解
            low_freq, high_freq = self.wavelet_extractors[s](x_stage)

            # 融合高频特征
            x_stage = x_stage + self.mdaf_modules[s](x_stage, high_freq)+self.mdaf_modules[s](x_stage, low_freq)
            # # 融合低频特征
            # x_stage = x_stage + self.mdaf_modules[s](x_stage, low_freq)

            # 分割输出
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x_stage))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x_stage))

            lres_input = x_stage

        # 反转分割输出
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        计算卷积特征图大小（主要用于内存估算，训练中不会调用）
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)

            # MDAF模块的计算量（如果启用）
            if self.use_mdaf_fusion and self.mdaf_modules is not None and s < len(self.mdaf_modules):
                # 估算MDAF的计算量
                channels = self.encoder.output_channels[-(s + 2)]
                spatial_size = np.prod(skip_sizes[-(s + 1)])
                # 3x3, 5x5, 7x7卷积的计算量
                mdaf_flops = channels * spatial_size * (3 ** 3 + 5 ** 3 + 7 ** 3 + 1) * 2  # 乘以2因为有两个输入
                output += mdaf_flops

        return output