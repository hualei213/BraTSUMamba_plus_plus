from typing import List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_AdM._Hilbert_Interleave import Mamba
from dynamic_network_architectures.building_blocks.MDAF import MDAF


# ==============================================================================
# 1) SAMF & LGPC support modules
# ==============================================================================
class RegionAwareModalFusion(nn.Module):
    def __init__(
            self,
            num_modalities: int = 4,
            tau_mod: float = 0.5,
            tau_reg: float = 1.0,
            sharpen_gamma: float = 1.0,
            residual: bool = True,
            alpha_init: float = 1.0,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.tau_mod = tau_mod
        self.tau_reg = tau_reg
        self.sharpen_gamma = sharpen_gamma
        self.residual = residual
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, feat_list: List[torch.Tensor], logit_list: List[torch.Tensor], return_maps: bool = False):
        fm = torch.stack(feat_list, dim=1)      # [B, M, C, D, H, W]
        sm = torch.stack(logit_list, dim=1)     # [B, M, R, D, H, W]

        w_m_given_r = F.softmax(sm / self.tau_mod, dim=1)
        sm_mean = sm.mean(dim=1)
        p_r = F.softmax(sm_mean / self.tau_reg, dim=1)
        w_m = (w_m_given_r * p_r.unsqueeze(1)).sum(dim=2)  # [B, M, D, H, W]

        if self.sharpen_gamma != 1.0:
            w_m = torch.clamp(w_m, min=self.eps) ** self.sharpen_gamma
            w_m = w_m / (w_m.sum(dim=1, keepdim=True) + self.eps)

        weighted = (w_m.unsqueeze(2) * fm).sum(dim=1)

        if self.residual:
            base = fm.mean(dim=1)
            fused = base + self.alpha * weighted
        else:
            fused = weighted

        if return_maps:
            return fused, w_m, p_r
        return fused, w_m


# ==============================================================================
# 2) AR-TPFE: tri-plane Laplacian frequency refinement
# ==============================================================================
class LaplacianPyramid2D(nn.Module):
    def __init__(
            self,
            levels: int = 1,
            initial_sigma: float = 1.0,
            kernel_size: int = 5,
            trainable_sigma: bool = True,
    ):
        super().__init__()
        self.levels = max(1, int(levels))
        self.kernel_size = kernel_size

        log_sigma = torch.log(torch.tensor(initial_sigma, dtype=torch.float32))
        if trainable_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def _create_gaussian_kernel_2d(self, sigma: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = torch.arange(self.kernel_size, device=device).float() - self.kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
        return kernel_2d

    def _gaussian_blur_2d(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        kernel_2d = self._create_gaussian_kernel_2d(self.sigma, x.device).to(dtype=x.dtype)
        kh, kw = kernel_2d.shape[-2], kernel_2d.shape[-1]
        weight = kernel_2d.expand(c, 1, kh, kw).contiguous()
        return F.conv2d(x, weight, bias=None, stride=1, padding=(kh // 2, kw // 2), groups=c)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current = x
        for _ in range(self.levels):
            if min(current.shape[-2:]) < 2:
                break
            blurred = self._gaussian_blur_2d(current)
            current = F.avg_pool2d(blurred, kernel_size=2, stride=2, ceil_mode=False)

        low_freq = F.interpolate(current, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high_freq = x - low_freq
        return low_freq, high_freq


class MultiPlaneLaplacianFeatureExtractor(nn.Module):
    def __init__(self, laplacian_level: int = 1, frequency_sigma: float = 1.0, trainable_frequency: bool = True):
        super().__init__()
        self.axial_laplacian = LaplacianPyramid2D(laplacian_level, frequency_sigma, 5, trainable_frequency)
        self.coronal_laplacian = LaplacianPyramid2D(laplacian_level, frequency_sigma, 5, trainable_frequency)
        self.sagittal_laplacian = LaplacianPyramid2D(laplacian_level, frequency_sigma, 5, trainable_frequency)

    def extract_2d_features(self, x: torch.Tensor, plane: str) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, d, h, w = x.shape

        if plane == "axial":
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * d, c, h, w)
            low, high = self.axial_laplacian(x_2d)
            low = low.view(b, d, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            high = high.view(b, d, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        elif plane == "coronal":
            x_2d = x.permute(0, 3, 1, 2, 4).contiguous().reshape(b * h, c, d, w)
            low, high = self.coronal_laplacian(x_2d)
            low = low.view(b, h, c, d, w).permute(0, 2, 3, 1, 4).contiguous()
            high = high.view(b, h, c, d, w).permute(0, 2, 3, 1, 4).contiguous()
        elif plane == "sagittal":
            x_2d = x.permute(0, 4, 1, 2, 3).contiguous().reshape(b * w, c, d, h)
            low, high = self.sagittal_laplacian(x_2d)
            low = low.view(b, w, c, d, h).permute(0, 2, 3, 4, 1).contiguous()
            high = high.view(b, w, c, d, h).permute(0, 2, 3, 4, 1).contiguous()
        else:
            raise ValueError(f"Unsupported plane: {plane}")

        return low, high

    def forward(self, x: torch.Tensor):
        low_axial, high_axial = self.extract_2d_features(x, "axial")
        low_coronal, high_coronal = self.extract_2d_features(x, "coronal")
        low_sagittal, high_sagittal = self.extract_2d_features(x, "sagittal")
        return [low_axial, low_coronal, low_sagittal], [high_axial, high_coronal, high_sagittal]


class LowFrequencyChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_mlp(self.avg_pool(x))


class HighFrequencySpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.activation(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class AdaptiveTriPlaneFrequencyRefinement(nn.Module):
    def __init__(
            self,
            channels: int,
            laplacian_level: int = 1,
            low_freq_reduction: int = 8,
            spatial_attention_kernel_size: int = 7,
    ):
        super().__init__()
        self.extractor = MultiPlaneLaplacianFeatureExtractor(laplacian_level=laplacian_level)
        self.low_freq_attention = LowFrequencyChannelAttention(channels, low_freq_reduction)
        self.high_freq_attention = HighFrequencySpatialAttention(spatial_attention_kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_features, high_features = self.extractor(x)
        refined_low = [self.low_freq_attention(feat) for feat in low_features]
        refined_high = [self.high_freq_attention(feat) for feat in high_features]
        low_freq = refined_low[0] + refined_low[1] + refined_low[2]
        high_freq = refined_high[0] + refined_high[1] + refined_high[2]
        return low_freq, high_freq


# ==============================================================================
# 3) 5-stage AdM encoder components
# ==============================================================================
class FeatureFusion3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_features = self.conv2(self.relu1(self.conv1(self.global_pool(x))))
        spatial_features = self.conv4(self.relu2(self.conv3(x)))
        return self.sigmoid(global_features + spatial_features)


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AFF_MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_slices_small=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="Bi",
            nslices_small=num_slices_small,
            coarse_mode="slice_hilbert_interleave",
        )
        self.fusion = FeatureFusion3D(channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        img_dims = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        x_skip = x
        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_sq, x_slice = self.mamba(x_norm)

        out_sq = x_sq.transpose(-1, -2).reshape(b, c, *img_dims)
        out_slice = x_slice.transpose(-1, -2).reshape(b, c, *img_dims)
        weight = self.fusion(out_sq + out_slice)
        out = out_sq * weight + (1 - weight) * out_slice
        return out + x_skip


class MSC(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv3d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv3d(channels, channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv3d(channels, channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(channels)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.relu(self.norm(self.proj(x)))
        x1 = self.relu2(self.norm2(self.proj2(x1)))
        x2 = self.relu3(self.norm3(self.proj3(x)))
        x = x1 * x2
        x = self.relu4(self.norm4(self.proj4(x)))
        return x + residual


class BiGranularityMambaEncoder(nn.Module):
    """
    Shared single-modality 5-stage Hilbert-BGM/AdM encoder.

    Default input [B, 1, 128, 128, 128] gives:
        stage 0: [B,  16, 64, 64, 64]
        stage 1: [B,  32, 32, 32, 32]
        stage 2: [B,  64, 16, 16, 16]
        stage 3: [B, 128,  8,  8,  8]
        stage 4: [B, 256,  4,  4,  4]
    """
    def __init__(
            self,
            in_chans: int = 1,
            depths: Sequence[int] = (2, 2, 2, 2, 2),
            dims: Sequence[int] = (16, 32, 64, 128, 256),
            out_indices: Sequence[int] = None,
    ):
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)
        self.num_stages = len(self.dims)

        assert len(self.depths) == self.num_stages, \
            f"depths and dims must have the same length, got {len(self.depths)} and {self.num_stages}."

        if out_indices is None:
            out_indices = tuple(range(self.num_stages))
        self.out_indices = set(out_indices)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(nn.Conv3d(in_chans, self.dims[0], kernel_size=7, stride=2, padding=3))
        )
        for i in range(self.num_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.InstanceNorm3d(self.dims[i]),
                nn.Conv3d(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2),
            ))

        default_slice_nums = [64, 32, 16, 8, 4]
        while len(default_slice_nums) < self.num_stages:
            default_slice_nums.append(max(1, default_slice_nums[-1] // 2))

        self.mscs = nn.ModuleList([MSC(ch) for ch in self.dims])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                AFF_MambaLayer(dim=self.dims[i], num_slices_small=default_slice_nums[i])
                for _ in range(self.depths[i])
            ])
            for i in range(self.num_stages)
        ])

        self.norms = nn.ModuleList([nn.InstanceNorm3d(ch) for ch in self.dims])
        self.mlps = nn.ModuleList([MlpChannel(ch, 2 * ch) for ch in self.dims])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.mscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(self.mlps[i](self.norms[i](x)))
        return tuple(outs)


# Backward-compatible alias for older import code.
Bi_granularity_Mamba = BiGranularityMambaEncoder


# ==============================================================================
# 4) nnU-Net-style BraTS-UMamba++ encoder and decoder
# ==============================================================================
class BraTSUMambaEncoder(nn.Module):
    """
    nnU-Net-style encoder wrapper.

    Forward:
        input : [B, 4, D, H, W]
        output: fused skips ordered from high to low resolution
                [16, 32, 64, 128, 256] channels by default.
    """
    def __init__(
            self,
            input_channels: int = 4,
            num_modalities: int = 4,
            num_regions: int = 4,
            depths: Sequence[int] = (2, 2, 2, 2, 2),
            dims: Sequence[int] = (16, 32, 64, 128, 256),
            collect_proto_info: bool = True,
    ):
        super().__init__()
        assert input_channels == num_modalities, "当前实现默认每个模态对应一个输入通道。"
        assert len(depths) == len(dims), "depths and dims must have the same length."
        assert len(dims) == 5, "当前 decoder 默认使用 5 个 encoder stage: 16, 32, 64, 128, 256。"

        self.input_channels = input_channels
        self.num_modalities = num_modalities
        self.num_regions = num_regions
        self.output_channels = list(dims)
        self.collect_proto_info = collect_proto_info
        self.proto_info = None

        # Metadata similar to dynamic-network-architectures encoders.
        self.strides = [(2, 2, 2)] * len(self.output_channels)
        self.kernel_sizes = [(3, 3, 3)] * len(self.output_channels)
        self.conv_op = nn.Conv3d
        self.conv_bias = False
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {"eps": 1e-5, "affine": True}
        self.dropout_op = None
        self.dropout_op_kwargs = None
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

        self.vit = BiGranularityMambaEncoder(
            in_chans=1,
            depths=depths,
            dims=dims,
            out_indices=tuple(range(len(dims))),
        )

        self.fusers = nn.ModuleList([
            RegionAwareModalFusion(num_modalities=num_modalities)
            for _ in self.output_channels
        ])
        self.region_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Conv3d(ch, num_regions, kernel_size=1, bias=True)
                for _ in range(num_modalities)
            ])
            for ch in self.output_channels
        ])
        self.proto_projectors = nn.ModuleList([
            nn.ModuleList([
                nn.Conv3d(ch, ch, kernel_size=1, bias=False)
                for _ in range(num_modalities)
            ])
            for ch in self.output_channels
        ])

    def forward_single_modality(self, x_modality: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.vit(x_modality)
        assert len(features) == len(self.output_channels), \
            f"Expected {len(self.output_channels)} AdM outputs, but got {len(features)}."
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        modality_inputs = [x[:, i:i + 1] for i in range(self.num_modalities)]
        modality_features = [self.forward_single_modality(modality_inputs[m]) for m in range(self.num_modalities)]

        fused_skips = []
        proto_info = []
        need_proto = self.training and self.collect_proto_info

        for s, _ in enumerate(self.output_channels):
            feat_list = [modality_features[m][s] for m in range(self.num_modalities)]
            logit_list = [self.region_heads[s][m](feat_list[m]) for m in range(self.num_modalities)]
            fused_feature, modality_weight = self.fusers[s](feat_list, logit_list, return_maps=False)
            fused_skips.append(fused_feature)

            if need_proto:
                proj_feats = [self.proto_projectors[s][m](feat_list[m]) for m in range(self.num_modalities)]
                proto_info.append({
                    "proj_feats": torch.stack(proj_feats, dim=1),  # [B, M, C, D, H, W]
                    "weights": modality_weight,                    # [B, M, D, H, W]
                })

        self.proto_info = proto_info if need_proto else None
        return fused_skips

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        """
        Lightweight nnU-Net-compatible feature-map estimate for the encoder.

        This estimate is intentionally conservative and is not an exact complexity count
        for Mamba/SAMF/LGPC. It mainly preserves compatibility with nnU-Net-style
        utilities that call network.compute_conv_feature_map_size(input_size).
        """
        input_size = tuple(int(i) for i in input_size)
        spatial_size = input_size
        output = np.int64(0)
        for ch in self.output_channels:
            spatial_size = tuple(max(1, i // 2) for i in spatial_size)
            # shared AdM feature, SAMF fused feature, region logits, and optional prototype projection proxy
            output += np.prod([ch, *spatial_size], dtype=np.int64)
            output += np.prod([ch, *spatial_size], dtype=np.int64)
            output += np.prod([self.num_regions, *spatial_size], dtype=np.int64)
        return output


class BraTSUMambaDecoder(nn.Module):
    """
    nnU-Net-style decoder with ConvTranspose3d upsampling and deep supervision.

    Compared with the vanilla nnU-Net decoder, the skip-fusion stage is kept as:
        ConvTranspose3d -> concat skip -> 1x1x1 fusion -> AR-TPFE/MDAF

    Deep supervision follows nnU-Net convention:
        - all segmentation heads are always built;
        - when deep_supervision=True, return a list ordered from high to low resolution;
        - when deep_supervision=False, return only the highest-resolution logits.
    """
    def __init__(
            self,
            num_classes: int = 4,
            encoder_channels: Sequence[int] = (16, 32, 64, 128, 256),
            final_decoder_channels: int = 8,
            deep_supervision: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_channels = list(encoder_channels)
        self.final_decoder_channels = final_decoder_channels
        self.deep_supervision = deep_supervision

        assert len(self.encoder_channels) == 5, "当前 decoder 需要 5 个 encoder stage。"

        # Four skip-fusion stages:
        #   256 -> 128, 128 -> 64, 64 -> 32, 32 -> 16
        self.transpconvs = nn.ModuleList()
        for in_ch, out_ch in zip(self.encoder_channels[:0:-1], self.encoder_channels[-2::-1]):
            self.transpconvs.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False))

        # Final no-skip upsampling stage:
        #   16 -> 8, 64^3 -> 128^3 for a 128^3 input.
        self.transpconvs.append(
            nn.ConvTranspose3d(self.encoder_channels[0], final_decoder_channels, kernel_size=2, stride=2, bias=False)
        )

        fusion_channels = self.encoder_channels[-2::-1]  # [128, 64, 32, 16]
        self.skip_fusion_convs = nn.ModuleList([
            nn.Conv3d(ch * 2, ch, kernel_size=1, stride=1, padding=0)
            for ch in fusion_channels
        ])
        self.frequency_refiners = nn.ModuleList([
            AdaptiveTriPlaneFrequencyRefinement(channels=ch)
            for ch in fusion_channels
        ])
        self.mdafs = nn.ModuleList([
            MDAF(ch, 8, "WithBias")
            for ch in fusion_channels
        ])

        # Seg heads are always built, following nnU-Net's deep-supervision convention.
        # They correspond to resolutions: 8^3, 16^3, 32^3, 64^3, and final 128^3.
        self.seg_layers = nn.ModuleList([
            nn.Conv3d(ch, num_classes, kernel_size=1)
            for ch in fusion_channels
        ])
        self.seg_layers.append(nn.Conv3d(final_decoder_channels, num_classes, kernel_size=1))

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x

    @staticmethod
    def _frequency_enhance(x: torch.Tensor, frequency_refiner: nn.Module, mdaf: nn.Module) -> torch.Tensor:
        low_f, high_f = frequency_refiner(x)
        return x + mdaf(x, high_f) + mdaf(x, low_f)

    def forward(self, skips: Sequence[torch.Tensor]):
        assert len(skips) == 5, f"BraTSUMambaDecoder expects 5 skip tensors, but got {len(skips)}."

        x = skips[-1]
        skip_targets = list(skips[-2::-1])  # [stage3, stage2, stage1, stage0]
        seg_outputs = []

        for s, skip in enumerate(skip_targets):
            x = self.transpconvs[s](x)
            x = self._match_size(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusion_convs[s](x)
            x = self._frequency_enhance(x, self.frequency_refiners[s], self.mdafs[s])

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))

        x = self.transpconvs[-1](x)
        logits = self.seg_layers[-1](x)

        if self.deep_supervision:
            seg_outputs.append(logits)
            # nnU-Net convention: largest output first.
            return seg_outputs[::-1]
        return logits

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        """
        Lightweight nnU-Net-compatible feature-map estimate.

        This is not an exact FLOPs counter for Mamba/MDAF/AR-TPFE. It is provided so that
        nnU-Net-style code paths expecting this method can run.
        """
        input_size = tuple(int(i) for i in input_size)
        skip_sizes = []
        cur = input_size
        for _ in range(len(self.encoder_channels)):
            cur = tuple(max(1, i // 2) for i in cur)
            skip_sizes.append(cur)

        output = np.int64(0)
        # Skip-fusion decoder stages output at skip_sizes[-2], ..., skip_sizes[0].
        decoder_sizes = skip_sizes[-2::-1]
        fusion_channels = self.encoder_channels[-2::-1]
        for s, (spatial_size, ch) in enumerate(zip(decoder_sizes, fusion_channels)):
            output += np.prod([ch, *spatial_size], dtype=np.int64)       # transposed-conv output
            output += np.prod([2 * ch, *spatial_size], dtype=np.int64)   # concat/fusion input proxy
            output += np.prod([ch, *spatial_size], dtype=np.int64)       # fusion output proxy
            if self.deep_supervision:
                output += np.prod([self.num_classes, *spatial_size], dtype=np.int64)

        # Final no-skip upsampling to original resolution.
        output += np.prod([self.final_decoder_channels, *input_size], dtype=np.int64)
        output += np.prod([self.num_classes, *input_size], dtype=np.int64)
        return output




class BraTSUMamba_plus_plus(nn.Module):
    """
    PlainConvUNet-compatible wrapper for nnU-Net-style integration.

    It accepts the same leading constructor arguments as dynamic_network_architectures'
    PlainConvUNet, but internally replaces the plain encoder/decoder with:
        - BraTSUMambaEncoder: shared 5-stage AdM/Mamba encoder + SAMF/LGPC support
        - BraTSUMambaDecoder: ConvTranspose3d decoder + AR-TPFE/MDAF + nnU-Net deep supervision

    Notes:
        1) The current implementation is 3D only.
        2) The nnU-Net planning arguments are accepted for interface compatibility.
           The AdM channels default to dims=(16, 32, 64, 128, 256) to preserve this version.
        3) Keep return_proto_info=False when using the default nnU-Net trainer, because nnU-Net
           expects logits or a list of logits, not (logits, proto_info).
    """
    def __init__(
            self,
            input_channels: int,
            n_stages: int,
            features_per_stage: Union[int, List[int], Tuple[int, ...]],
            conv_op: Type[nn.Module],
            kernel_sizes: Union[int, List[int], Tuple[int, ...]],
            strides: Union[int, List[int], Tuple[int, ...]],
            n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
            num_classes: int,
            n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
            conv_bias: bool = False,
            norm_op: Optional[Type[nn.Module]] = None,
            norm_op_kwargs: Optional[dict] = None,
            dropout_op: Optional[Type[nn.Module]] = None,
            dropout_op_kwargs: Optional[dict] = None,
            nonlin: Optional[Type[torch.nn.Module]] = None,
            nonlin_kwargs: Optional[dict] = None,
            deep_supervision: bool = False,
            nonlin_first: bool = False,
            # BraTS-UMamba++ specific options
            num_modalities: int = 4,
            num_regions: int = 4,
            depths: Sequence[int] = (2, 2, 2, 2, 2),
            dims: Sequence[int] = (16, 32, 64, 128, 256),
            final_decoder_channels: int = 8,
            return_proto_info: bool = False,
            **kwargs,
    ):
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError("当前 BraTSUMamba_plus_plus 只支持 nn.Conv3d / 3D segmentation。")
        if input_channels != num_modalities:
            raise ValueError(
                f"input_channels ({input_channels}) must equal num_modalities ({num_modalities}) "
                "because each MRI modality is processed as one channel."
            )

        # Keep these attributes so nnU-Net-style utilities can inspect them.
        self.input_channels = input_channels
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_conv_per_stage = n_conv_per_stage
        self.num_classes = num_classes
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.deep_supervision = deep_supervision
        self.nonlin_first = nonlin_first
        self.return_proto_info = return_proto_info

        self.encoder = BraTSUMambaEncoder(
            input_channels=input_channels,
            num_modalities=num_modalities,
            num_regions=num_regions,
            depths=depths,
            dims=dims,
            collect_proto_info=return_proto_info,
        )
        self.decoder = BraTSUMambaDecoder(
            num_classes=num_classes,
            encoder_channels=self.encoder.output_channels,
            final_decoder_channels=final_decoder_channels,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        out = self.decoder(skips)
        if self.training and self.return_proto_info:
            return out, self.encoder.proto_info
        return out

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        assert len(input_size) == 3, "当前 BraTSUMamba_plus_plus 只支持 3D 输入尺寸。"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module: nn.Module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
            if getattr(module, "weight", None) is not None:
                nn.init.constant_(module.weight, 1)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)



# ==============================================================================
# 7) LGPC loss: label-guided prototype consistency
# ==============================================================================
class LabelGuidedPrototypeLoss(nn.Module):
    """
    Label-Guided Prototype Consistency (LGPC) loss.

    This implementation follows the previous BraTS-UMamba++ prototype strategy:
      1) collect projected modality features from each encoder stage:
             proj_feats: [B, M, C, D, H, W]
      2) collect SAMF modality weights:
             weights:    [B, M, D, H, W]
      3) use the ground-truth label map to compute class-wise, modality-wise
         weighted prototypes for each sample and each scale;
      4) align same-class prototypes across modalities;
      5) separate different-class prototypes within each modality.

    Notes:
      - By default, background class 0 is ignored.
      - No global memory bank is used; prototypes are computed per mini-batch/sample.
      - The target can be either a single tensor [B,1,D,H,W] or a deep-supervision
        target list/tuple. The loss automatically matches the target resolution to
        each proto_info stage.
    """
    def __init__(
            self,
            num_classes: int = 4,
            ignore_index: Optional[int] = 0,
            delta_pos: float = 0.9,
            delta_neg: float = 0.1,
            momentum: float = 0.9,
            lambda_align: float = 1.0,
            lambda_sep: float = 1.0,
            eps: float = 1e-6,
            min_pixels: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.momentum = momentum  # kept only for old interface compatibility
        self.lambda_align = lambda_align
        self.lambda_sep = lambda_sep
        self.eps = eps
        self.min_pixels = min_pixels

    @staticmethod
    def _as_label_tensor(target: torch.Tensor) -> torch.Tensor:
        """Convert target to [B,D,H,W] long tensor when possible."""
        if target.ndim == 5:
            target = target.squeeze(1)
        return target.long()

    def _get_target_for_shape(
            self,
            target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
            current_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Fetch or resize the segmentation target for the current proto_info scale.
        Supports both a single target and nnU-Net-style deep-supervision target lists.
        """
        if isinstance(target, (list, tuple)):
            highest_res_target = target[0]
            target_map = {tuple(t.shape[2:]): t for t in target if t.ndim == 5}
            if current_shape in target_map:
                curr_target = target_map[current_shape]
            else:
                curr_target = F.interpolate(highest_res_target.float(), size=current_shape, mode="nearest")
        else:
            highest_res_target = target
            if target.ndim == 5 and tuple(target.shape[2:]) == current_shape:
                curr_target = target
            else:
                if target.ndim == 4:
                    highest_res_target = target.unsqueeze(1)
                curr_target = F.interpolate(highest_res_target.float(), size=current_shape, mode="nearest")

        return self._as_label_tensor(curr_target)

    def forward(
            self,
            proto_info: Optional[List[dict]],
            target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        if proto_info is None or len(proto_info) == 0:
            if isinstance(target, (list, tuple)):
                return target[0].new_zeros((), dtype=torch.float32)
            return target.new_zeros((), dtype=torch.float32)

        loss_align = None
        loss_sep = None
        cnt_align = 0
        cnt_sep = 0

        for info in proto_info:
            feats = info["proj_feats"]   # [B, M, C, D, H, W]
            weights = info["weights"]    # [B, M, D, H, W]

            if loss_align is None:
                loss_align = feats.new_zeros(())
                loss_sep = feats.new_zeros(())

            bsz, num_mods, channels, depth, height, width = feats.shape
            current_shape = (depth, height, width)
            curr_target = self._get_target_for_shape(target, current_shape)  # [B,D,H,W]

            stage_prototypes = {}  # key: (b, m, k), value: (omega, proto[C])

            # ------------------------------------------------------------------
            # 1) Compute label-guided modality prototypes.
            # ------------------------------------------------------------------
            for b in range(bsz):
                for m in range(num_mods):
                    feat_bm = feats[b, m]       # [C,D,H,W]
                    weight_bm = weights[b, m]   # [D,H,W]

                    for k in range(self.num_classes):
                        if self.ignore_index is not None and k == self.ignore_index:
                            continue

                        mask = curr_target[b] == k  # [D,H,W]
                        pixel_count = mask.sum()
                        if pixel_count.item() < self.min_pixels:
                            continue

                        mask_f = mask.to(dtype=feat_bm.dtype)
                        weighted_mask = weight_bm * mask_f
                        total_weight = weighted_mask.sum() + self.eps

                        # omega measures the average SAMF confidence of modality m in class k.
                        omega = total_weight / (pixel_count.to(dtype=feat_bm.dtype) + self.eps)

                        proto = (feat_bm * weighted_mask.unsqueeze(0)).sum(dim=(1, 2, 3)) / total_weight
                        proto = F.normalize(proto, p=2, dim=0)
                        stage_prototypes[(b, m, k)] = (omega, proto)

            # ------------------------------------------------------------------
            # 2) Cross-modality alignment for the same class.
            #    Same-region prototypes from different modalities are encouraged
            #    to have cosine similarity larger than delta_pos.
            # ------------------------------------------------------------------
            for b in range(bsz):
                for k in range(self.num_classes):
                    if self.ignore_index is not None and k == self.ignore_index:
                        continue

                    valid_mods = [m for m in range(num_mods) if (b, m, k) in stage_prototypes]
                    if len(valid_mods) < 2:
                        continue

                    for i in range(len(valid_mods)):
                        for j in range(i + 1, len(valid_mods)):
                            m_i, m_j = valid_mods[i], valid_mods[j]
                            omega_i, proto_i = stage_prototypes[(b, m_i, k)]
                            omega_j, proto_j = stage_prototypes[(b, m_j, k)]
                            sim = torch.dot(proto_i, proto_j)
                            loss_align = loss_align + (omega_i * omega_j) * F.relu(self.delta_pos - sim)
                            cnt_align += 1

            # ------------------------------------------------------------------
            # 3) Intra-modality separation for different classes.
            #    Different-region prototypes within the same modality are pushed
            #    below delta_neg.
            # ------------------------------------------------------------------
            for b in range(bsz):
                for m in range(num_mods):
                    valid_classes = [
                        k for k in range(self.num_classes)
                        if (self.ignore_index is None or k != self.ignore_index)
                        and (b, m, k) in stage_prototypes
                    ]
                    if len(valid_classes) < 2:
                        continue

                    for i in range(len(valid_classes)):
                        for j in range(i + 1, len(valid_classes)):
                            k_i, k_j = valid_classes[i], valid_classes[j]
                            _, proto_i = stage_prototypes[(b, m, k_i)]
                            _, proto_j = stage_prototypes[(b, m, k_j)]
                            sim = torch.dot(proto_i, proto_j)
                            loss_sep = loss_sep + F.relu(sim - self.delta_neg)
                            cnt_sep += 1

        if loss_align is None:
            if isinstance(target, (list, tuple)):
                return target[0].new_zeros((), dtype=torch.float32)
            return target.new_zeros((), dtype=torch.float32)

        loss_align = loss_align / (cnt_align + self.eps)
        loss_sep = loss_sep / (cnt_sep + self.eps)
        return self.lambda_align * loss_align + self.lambda_sep * loss_sep


# Common aliases for project-side imports.
BraTSUMambaPlusPlus = BraTSUMamba_plus_plus
