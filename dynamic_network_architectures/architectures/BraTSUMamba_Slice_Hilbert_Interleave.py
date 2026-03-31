from typing import Union, Type, List, Tuple, Optional
import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from mamba_AdM._Hilbert_Interleave import Mamba


class FeatureFusion3D(nn.Module):
    """Minimal sigmoid gate for (out_sq, out_slice) fusion."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class MSC(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # 1×1
        self.proj1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.act1 = nn.ReLU(inplace=True)

        # 3×3
        self.proj3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.act3 = nn.ReLU(inplace=True)

        # 5×5
        self.proj5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.norm5 = nn.InstanceNorm3d(in_channels)
        self.act5 = nn.ReLU(inplace=True)

        # 7×7
        self.proj7 = nn.Conv3d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.norm7 = nn.InstanceNorm3d(in_channels)
        self.act7 = nn.ReLU(inplace=True)

        # fuse 1×1
        self.proj_fuse = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm_fuse = nn.InstanceNorm3d(in_channels)
        self.act_fuse = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = x

        x1 = self.act1(self.norm1(self.proj1(x)))
        x3 = self.act3(self.norm3(self.proj3(x)))
        x5 = self.act5(self.norm5(self.proj5(x)))
        x7 = self.act7(self.norm7(self.proj7(x)))

        x = x1 + x3 + x5 + x7
        x = self.act_fuse(self.norm_fuse(self.proj_fuse(x)))
        return x + x_res



class AFF_MambaLayer(nn.Module):
    """
    Adaptive Feature Fusion + Bi-granularity Mamba.

    Input:  x [B,C,D,H,W]
    Output: same shape (residual)
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 num_slices_small: Optional[int] = None, coarse_mode: str = "slice_hilbert_interleave",):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="Bi",
            nslices_small=num_slices_small,
            coarse_mode=coarse_mode,
        )
        self.fusion = FeatureFusion3D(channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        assert C == self.dim
        x_skip = x

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # [B,C,D,H,W] -> [B,L,C]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # Mamba returns (out_sq, out_slice), both [B,L,C]
        x_sq, x_slice = self.mamba(x_norm)

        out_sq = x_sq.transpose(-1, -2).reshape(B, C, *img_dims)
        out_slice = x_slice.transpose(-1, -2).reshape(B, C, *img_dims)

        # gating fusion
        w = self.fusion(out_sq + out_slice)  # [B,C,D,H,W] in (0,1)
        out = out_sq * w + (1.0 - w) * out_slice

        return out + x_skip


class PlainConvUNet(nn.Module):
    """
    PlainConvUNet + per-skip (MSC + BGM) enhancement.

    Modifications:
    - Added `bgm_start_stage` parameter to skip high-resolution stages (e.g., stage 0)
      to save memory/compute.
    """

    def __init__(
            self,
            input_channels: int,
            n_stages: int,
            features_per_stage: Union[int, List[int], Tuple[int, ...]],
            conv_op: Type[_ConvNd],
            kernel_sizes: Union[int, List[int], Tuple[int, ...]],
            strides: Union[int, List[int], Tuple[int, ...]],
            n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
            num_classes: int,
            n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
            bgm_depths: list = [2, 2, 2, 2, 2, 2],
            num_slices_small_list: list = [None, 64, 32, 16, 8, 4],
            conv_bias: bool = False,
            norm_op: Union[None, Type[nn.Module]] = None,
            norm_op_kwargs: dict = None,
            dropout_op: Union[None, Type[_DropoutNd]] = None,
            dropout_op_kwargs: dict = None,
            nonlin: Union[None, Type[torch.nn.Module]] = None,
            nonlin_kwargs: dict = None,
            deep_supervision: bool = False,
            nonlin_first: bool = False,
            bgm_start_stage: int = 1,  # <--- [新增参数] 默认为 1，即跳过 Stage 0 (128x128x128)
    ):
        super().__init__()

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages

        # if bgm_depths is None:
        #     bgm_depths = [1] * n_stages
        # assert len(bgm_depths) == n_stages
        #
        # if num_slices_small_list is None:
        #     num_slices_small_list = [None] * n_stages
        # assert len(num_slices_small_list) == n_stages

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        # enhancement blocks per skip
        self.mscs = nn.ModuleList()
        self.bgms = nn.ModuleList()

        for i in range(n_stages):
            # [逻辑修改] 如果当前阶段小于起始阶段，使用 Identity (不做任何处理，直接通过)
            if i < bgm_start_stage:
                self.mscs.append(nn.Identity())
                self.bgms.append(nn.Identity())
            else:
                # 否则，构建 MSC 和 BGM 模块
                self.mscs.append(MSC(int(features_per_stage[i])))
                self.bgms.append(
                    nn.Sequential(*[
                        AFF_MambaLayer(dim=int(features_per_stage[i]), num_slices_small=num_slices_small_list[i])
                        for _ in range(bgm_depths[i])
                    ])
                )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)  # list of tensors at multiple resolutions

        # Apply MSC + BGM on each skip
        # 对于 i < bgm_start_stage 的部分，bgms[i] 和 mscs[i] 是 Identity，不增加计算量
        skips = [self.bgms[i](self.mscs[i](skips[i])) for i in range(len(skips))]

        # print(f"\n--- DEBUG: Skips Shapes (Batch: {x.shape[0]}) ---")
        # for i, s in enumerate(skips):
        #     print(f"  Stage {i}: {s.shape}")
        # print("------------------------------------------\n")

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)






if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

