from typing import Union, Type, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder_bratsumamba_tri import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

# AdM / BGM (Bi-granularity Mamba)
from mamba_AdM._Hilbert_Interleave import Mamba


# ==============================================================================
# 1) SAMF (Region-aware Modal Fusion)  [unchanged]
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
            eps: float = 1e-6
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.tau_mod = tau_mod
        self.tau_reg = tau_reg
        self.sharpen_gamma = sharpen_gamma
        self.residual = residual
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, feat_list, logit_list, return_maps: bool = False):
        """
        feat_list:  list of [B,C,D,H,W], length M
        logit_list: list of [B,R,D,H,W], length M
        """
        Fm = torch.stack(feat_list, dim=1)  # [B,M,C,D,H,W]
        Sm = torch.stack(logit_list, dim=1)  # [B,M,R,D,H,W]

        # Step 1: modality weights given region
        w_m_given_r = F.softmax(Sm / self.tau_mod, dim=1)  # [B,M,R,D,H,W]

        # Step 2: region prior
        Sm_mean = Sm.mean(dim=1)  # [B,R,D,H,W]
        P_r = F.softmax(Sm_mean / self.tau_reg, dim=1)  # [B,R,D,H,W]

        # Step 3: integrate over regions -> modality weights per voxel
        w_m = (w_m_given_r * P_r.unsqueeze(1)).sum(dim=2)  # [B,M,D,H,W]

        # Optional sharpen
        if self.sharpen_gamma != 1.0:
            w_m = torch.clamp(w_m, min=self.eps) ** self.sharpen_gamma
            w_m = w_m / (w_m.sum(dim=1, keepdim=True) + self.eps)

        # Step 4: fuse
        weighted = (w_m.unsqueeze(2) * Fm).sum(dim=1)  # [B,C,D,H,W]

        if self.residual:
            base = Fm.mean(dim=1)
            F_fuse = base + self.alpha * weighted
        else:
            F_fuse = weighted

        if return_maps:
            return F_fuse, w_m, P_r

        # default returns weights for LGPC loss
        return F_fuse, w_m


# ==============================================================================
# 2) AdM blocks: MSC + Bi-granularity Mamba (Fine: full 3D Hilbert; Coarse: slice-Hilbert-interleave)
# ==============================================================================
class FeatureFusion3D(nn.Module):
    """Minimal sigmoid gate for (out_sq, out_slice) fusion."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class MSC(nn.Module):
    """
    Multi-scale convolution block (1x1 + 3x3 + 5x5 + 7x7, then fuse).
    Keeps shape & channels.
    """
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
    Input/Output: [B,C,D,H,W] (residual).
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices_small: Optional[int] = None,
        coarse_mode: str = "slice_hilbert_interleave",
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # Mamba will output two streams: fine (3D Hilbert) and coarse (slice-based interleave by default)
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
        assert C == self.dim, f"channel mismatch: got {C}, expect {self.dim}"
        x_skip = x

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # [B,C,D,H,W] -> [B,L,C]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # returns (x_sq, x_slice): both [B,L,C]
        x_sq, x_slice = self.mamba(x_norm)

        out_sq = x_sq.transpose(-1, -2).reshape(B, C, *img_dims)
        out_slice = x_slice.transpose(-1, -2).reshape(B, C, *img_dims)

        w = self.fusion(out_sq + out_slice)
        out = out_sq * w + (1.0 - w) * out_slice
        return out + x_skip


def _normalize_stage_list(val, n_stages: int, default_fill):
    """
    Make val a list of length n_stages.
    - None -> [default_fill]*n
    - scalar -> repeat
    - list/tuple -> pad/truncate
    """
    if val is None:
        return [default_fill] * n_stages
    if isinstance(val, (int, float)):
        return [val] * n_stages
    if isinstance(val, (list, tuple)):
        v = list(val)
        if len(v) < n_stages:
            v = v + [v[-1]] * (n_stages - len(v))
        elif len(v) > n_stages:
            v = v[:n_stages]
        return v
    raise TypeError(f"Unsupported type for stage list: {type(val)}")


# ==============================================================================
# 3) Network: Multi-modal encoders + (per-modality AdM) + SAMF(+LGPC) + decoder
# ==============================================================================
class PlainConvUNet(nn.Module):
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
            # --- SAMF / LGPC ---
            num_regions: int = 4,
            num_modalities: int = 4,
            tau_mod: float = 0.5,
            tau_reg: float = 1.0,
            sharpen_gamma: float = 1.0,
            residual_fusion: bool = True,
            # --- AdM (per-modality) ---
            bgm_depths: Optional[List[int]] = None,
            num_slices_small_list: Optional[List[Optional[int]]] = None,
            bgm_start_stage: int = 1,  # skip highest-res (stage0) to reduce compute
            # --- misc ---
            conv_bias: bool = False,
            norm_op: Union[None, Type[nn.Module]] = None,
            norm_op_kwargs: dict = None,
            dropout_op: Union[None, Type[_DropoutNd]] = None,
            dropout_op_kwargs: dict = None,
            nonlin: Union[None, Type[torch.nn.Module]] = None,
            nonlin_kwargs: dict = None,
            deep_supervision: bool = False,
            nonlin_first: bool = False
    ):
        super().__init__()

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        else:
            features_per_stage = list(features_per_stage)

        # defaults (safe for nnU-Net kwargs construction)
        # recommended for 128^3 patch with 6 stages: [None, 64, 32, 16, 8, 4]
        if num_slices_small_list is None:
            num_slices_small_list = [None, 64, 32, 16, 8, 4]
        bgm_depths = _normalize_stage_list(bgm_depths, n_stages, default_fill=2)
        num_slices_small_list = _normalize_stage_list(num_slices_small_list, n_stages, default_fill=None)

        self.num_modalities = num_modalities

        # --- modality-specific encoders ---
        self.down = nn.ModuleList([
            PlainConvEncoder(
                1, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                nonlin_first=nonlin_first
            ) for _ in range(num_modalities)
        ])

        self.template_down = self.down[0]

        # --- decoder (same as your v2, with tri-plane frequency fusion etc.) ---
        self.decoder = UNetDecoder(
            self.template_down, num_classes, n_conv_per_stage_decoder, deep_supervision,
            nonlin_first=nonlin_first, use_mdaf_fusion=True, use_frequency_decomposition=True,
            frequency_levels=1
        )

        # --- per-stage region heads (for SAMF) ---
        self.region_heads = nn.ModuleList([
            nn.ModuleList([
                conv_op(features_per_stage[s], num_regions, 1, bias=True)
                for _ in range(num_modalities)
            ]) for s in range(n_stages)
        ])

        self.fusers = nn.ModuleList([
            RegionAwareModalFusion(
                num_modalities=num_modalities,
                tau_mod=tau_mod,
                tau_reg=tau_reg,
                sharpen_gamma=sharpen_gamma,
                residual=residual_fusion
            ) for _ in range(n_stages)
        ])

        # --- prototype projector for LGPC loss ---
        self.proto_projectors = nn.ModuleList([
            nn.ModuleList([
                conv_op(features_per_stage[s], features_per_stage[s], 1, bias=False)
                for _ in range(num_modalities)
            ]) for s in range(n_stages)
        ])

        # --- NEW: per-modality AdM blocks (MSC + BGM) ---
        # shape: [stage][modality]
        self.mscs = nn.ModuleList()
        self.bgms = nn.ModuleList()
        for s in range(n_stages):
            msc_stage = nn.ModuleList()
            bgm_stage = nn.ModuleList()
            for _m in range(num_modalities):
                if s < bgm_start_stage:
                    msc_stage.append(nn.Identity())
                    bgm_stage.append(nn.Identity())
                else:
                    msc_stage.append(MSC(int(features_per_stage[s])))
                    bgm_stage.append(nn.Sequential(*[
                        AFF_MambaLayer(
                            dim=int(features_per_stage[s]),
                            num_slices_small=num_slices_small_list[s],
                            coarse_mode="slice_hilbert_interleave",
                        )
                        for _ in range(int(bgm_depths[s]))
                    ]))
            self.mscs.append(msc_stage)
            self.bgms.append(bgm_stage)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        # x: [B,4,D,H,W] -> list of 4 tensors [B,1,D,H,W]
        xs = [x[:, i:i + 1] for i in range(self.num_modalities)]
        all_skips = [enc(xm) for enc, xm in zip(self.down, xs)]  # list[M] of list[S]

        # --- NEW: per-modality AdM enhancement before SAMF fusion ---
        for m in range(self.num_modalities):
            for s in range(len(all_skips[m])):
                all_skips[m][s] = self.bgms[s][m](self.mscs[s][m](all_skips[m][s]))

        fused_skips = []
        proto_info = []

        for s in range(len(all_skips[0])):
            feat_list = [all_skips[m][s] for m in range(self.num_modalities)]
            logit_list = [self.region_heads[s][m](feat_list[m]) for m in range(self.num_modalities)]

            F_fuse, w_m = self.fusers[s](feat_list, logit_list, return_maps=False)
            fused_skips.append(F_fuse)

            # save proto info only in training to save memory
            if self.training:
                proj_feats = [self.proto_projectors[s][m](feat_list[m]) for m in range(self.num_modalities)]
                proj_feats_stack = torch.stack(proj_feats, dim=1)  # [B,M,C,D,H,W]
                proto_info.append({
                    "proj_feats": proj_feats_stack,
                    "weights": w_m
                })

        out = self.decoder(fused_skips)
        if self.training:
            return out, proto_info
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.template_encoder.conv_op
        ), "Give input_size=(x,y(,z)) only."
        return self.template_encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


# ==============================================================================
# 4) LGPC loss (unchanged)
# ==============================================================================
class LabelGuidedPrototypeLoss(nn.Module):
    """
    Prototype regularization:
      - Align: same class prototypes across modalities should be close
      - Sep:   different class prototypes within the same modality should be far

    V4: no global prototype memory, per-sample prototypes; default skip background class (ignore_index=0).
    """
    def __init__(
            self,
            num_classes: int = 4,
            ignore_index: int = 0,
            delta_pos: float = 0.9,
            delta_neg: float = 0.1,
            momentum: float = 0.9,  # deprecated (kept for old interface compatibility)
            lambda_align: float = 1.0,
            lambda_sep: float = 1.0,
            eps: float = 1e-6,
            min_pixels: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.momentum = momentum  # deprecated
        self.lambda_align = lambda_align
        self.lambda_sep = lambda_sep
        self.eps = eps
        self.min_pixels = min_pixels

    def forward(self, proto_info: List[dict], target: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]):
        # --- target mapping to multi-res ---
        target_map = {}
        highest_res_target = None
        if isinstance(target, (list, tuple)):
            for t in target:
                target_map[tuple(t.shape[2:])] = t
            highest_res_target = target[0]
        else:
            target_map[tuple(target.shape[2:])] = target
            highest_res_target = target

        loss_align = None
        loss_sep = None
        cnt_align = 0
        cnt_sep = 0

        for s, info in enumerate(proto_info):
            feats = info["proj_feats"]   # [B,M,C,D,H,W]
            weights = info["weights"]    # [B,M,D,H,W]
            device = feats.device

            if loss_align is None:
                loss_align = feats.new_zeros(())
                loss_sep = feats.new_zeros(())

            B, M, C, D, H, W = feats.shape
            current_shape = (D, H, W)

            if current_shape in target_map:
                curr_target = target_map[current_shape]
            else:
                curr_target = F.interpolate(highest_res_target.float(), size=current_shape, mode="nearest")

            if curr_target.ndim == 5:
                curr_target = curr_target.squeeze(1)
            curr_target = curr_target.long()  # [B,D,H,W]

            stage_prototypes = {}  # (b,m,k) -> (omega, proto[C])

            for b in range(B):
                for m in range(M):
                    feat_bm = feats[b, m]      # [C,D,H,W]
                    weight_bm = weights[b, m]  # [D,H,W]

                    for k in range(self.num_classes):
                        if (self.ignore_index is not None) and (k == self.ignore_index):
                            continue

                        mask = (curr_target[b] == k)  # [D,H,W]
                        pixel_count = mask.sum()
                        if pixel_count.item() < self.min_pixels:
                            continue

                        mask_f = mask.float()
                        weighted_mask = weight_bm * mask_f
                        total_weight = weighted_mask.sum() + self.eps
                        omega = total_weight / (pixel_count.float() + self.eps)

                        proto = (feat_bm * weighted_mask.unsqueeze(0)).sum(dim=(1, 2, 3)) / total_weight
                        proto = F.normalize(proto, p=2, dim=0)
                        stage_prototypes[(b, m, k)] = (omega, proto)

            # Align
            for b in range(B):
                for k in range(self.num_classes):
                    if (self.ignore_index is not None) and (k == self.ignore_index):
                        continue
                    valid_mods = [m for m in range(M) if (b, m, k) in stage_prototypes]
                    if len(valid_mods) < 2:
                        continue
                    for i in range(len(valid_mods)):
                        for j in range(i + 1, len(valid_mods)):
                            m, n = valid_mods[i], valid_mods[j]
                            omega_m, proto_m = stage_prototypes[(b, m, k)]
                            omega_n, proto_n = stage_prototypes[(b, n, k)]
                            sim = torch.dot(proto_m, proto_n)
                            loss_align = loss_align + (omega_m * omega_n) * F.relu(self.delta_pos - sim)
                            cnt_align += 1

            # Sep
            for b in range(B):
                for m in range(M):
                    valid_classes = [k for k in range(self.num_classes)
                                     if ((self.ignore_index is None) or (k != self.ignore_index))
                                     and ((b, m, k) in stage_prototypes)]
                    if len(valid_classes) < 2:
                        continue
                    for i in range(len(valid_classes)):
                        for j in range(i + 1, len(valid_classes)):
                            k1, k2 = valid_classes[i], valid_classes[j]
                            _, p1 = stage_prototypes[(b, m, k1)]
                            _, p2 = stage_prototypes[(b, m, k2)]
                            sim = torch.dot(p1, p2)
                            loss_sep = loss_sep + F.relu(sim - self.delta_neg)
                            cnt_sep += 1

        if loss_align is None:
            # proto_info empty
            if isinstance(target, (list, tuple)):
                return torch.tensor(0.0, device=target[0].device)
            return torch.tensor(0.0, device=target.device)

        loss_align = loss_align / (cnt_align + self.eps)
        loss_sep = loss_sep / (cnt_sep + self.eps)
        return self.lambda_align * loss_align + self.lambda_sep * loss_sep
