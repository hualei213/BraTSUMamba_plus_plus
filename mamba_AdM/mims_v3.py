import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Sequence

from mamba_AdM.BGM import Mamba_mims


# ---------------------------
# Per-modality embed
# ---------------------------
class PerModalityEmbed(nn.Module):
    def __init__(self, c0: int, embed_type: str = "conv", stride: int = 2):
        super().__init__()
        layers = [
            nn.Conv3d(1, c0, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.InstanceNorm3d(c0),
            nn.GELU(),
        ]
        if embed_type == "dsconv":
            layers += [
                nn.Conv3d(c0, c0, kernel_size=3, stride=1, padding=1, groups=c0, bias=False),
                nn.InstanceNorm3d(c0),
                nn.GELU(),
                nn.Conv3d(c0, c0, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm3d(c0),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        return self.net(x)


# ---------------------------
# MambaLayer3D: 对 (B,C,D,H,W) 做三向扫描 + residual
# ---------------------------
class MambaLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bimamba_type: str = "Bi",              # "Bi" or "none"
        directions: Sequence[str] = ("w", "h", "d"),
    ):
        super().__init__()
        self.dim = dim
        self.directions = tuple(directions)

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba_mims(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
        )

    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C = x.shape[:2]
        D, H, W = x.shape[2:]
        L = D * H * W

        # (B, C, D, H, W) -> (B, L, C)
        seq = x.reshape(B, C, L).transpose(-1, -2)
        seq = self.norm(seq)
        seq = self.mamba(seq)  # (B, L, C)

        # (B, L, C) -> (B, C, D, H, W)
        y = seq.transpose(-1, -2).reshape(B, C, D, H, W)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []

        # w-direction: (D,H,W) flatten => W 最快
        if "w" in self.directions:
            outs.append(self._mamba_forward(x))

        # h-direction: swap H<->W => flatten 时 H 最快
        if "h" in self.directions:
            x_h = rearrange(x, "b c d h w -> b c d w h")
            y_h = self._mamba_forward(x_h)
            y_h = rearrange(y_h, "b c d w h -> b c d h w")
            outs.append(y_h)

        # d-direction: rotate => flatten 时 D 最快
        if "d" in self.directions:
            x_d = rearrange(x, "b c d h w -> b c h w d")
            y_d = self._mamba_forward(x_d)
            y_d = rearrange(y_d, "b c h w d -> b c d h w")
            outs.append(y_d)

        if len(outs) == 0:
            return x

        return sum(outs) + x


# ---------------------------
#改进版MIMS: 独立 embed +intra +inter(long-token) + upsample
# ---------------------------
class MIMS(nn.Module):
    def __init__(
        self,
        out_dim: int = 32,
        c0: int = 16,
        embed_type: str = "conv",
        embed_stride: int = 2,
        directions: Sequence[str] = ("w", "h", "d"),
        mamba_kwargs: Optional[dict] = None,
        add_residual_from_embed: bool = True,

        enable_intra: bool = True,   # 是否保留你原先的 per-modality 扫描
        enable_inter: bool = True,   # 是否启用 long-token 模态间扫描
    ):
        super().__init__()
        self.embed_stride = embed_stride
        self.add_residual_from_embed = add_residual_from_embed
        self.enable_intra = enable_intra
        self.enable_inter = enable_inter

        if mamba_kwargs is None:
            mamba_kwargs = dict(d_state=16, d_conv=4, expand=2, bimamba_type="Bi")

        d_state = mamba_kwargs.get("d_state", 16)
        d_conv = mamba_kwargs.get("d_conv", 4)
        expand = mamba_kwargs.get("expand", 2)
        bimamba_type = mamba_kwargs.get("bimamba_type", "Bi")

        # 1) 四模态独立 embed
        self.embeds = nn.ModuleList(
            [PerModalityEmbed(c0=c0, embed_type=embed_type, stride=embed_stride) for _ in range(4)]
        )

        # 2) Intra: 每模态独立扫描 (dim=C0)
        if self.enable_intra:
            self.mamba_intra = MambaLayer3D(
                dim=c0,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
                directions=directions,
            )

        # 3) Inter(long-token): 同位置四模态拼成一个 token (dim=4*C0)
        #    再用 1x1x1 Conv 降回 C0，方便和其它分支融合
        if self.enable_inter:
            self.mamba_inter = MambaLayer3D(
                dim=4 * c0,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
                directions=directions,
            )
            self.inter_reduce = nn.Conv3d(4 * c0, c0, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.inter_reduce.weight, a=1e-2)

        # 4) 上采样恢复到原分辨率，并输出 out_dim 通道
        if embed_stride > 1:
            self.up_sample = nn.ConvTranspose3d(
                in_channels=c0,
                out_channels=out_dim,
                kernel_size=embed_stride,
                stride=embed_stride,
                bias=False,
            )
        else:
            self.up_sample = nn.Conv3d(c0, out_dim, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.up_sample.weight, a=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, 128, 128, 128)
        return: (B, out_dim, 128, 128, 128)
        """
        # split modalities
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]
        x4 = x[:, 3:4]

        # independent embed -> (B, C0, D', H', W')
        u1 = self.embeds[0](x1)
        u2 = self.embeds[1](x2)
        u3 = self.embeds[2](x3)
        u4 = self.embeds[3](x4)

        embed_sum = u1 + u2 + u3 + u4

        # Intra branch: per modality
        intra_sum = 0
        if self.enable_intra:
            intra_sum = (
                self.mamba_intra(u1)
                + self.mamba_intra(u2)
                + self.mamba_intra(u3)
                + self.mamba_intra(u4)
            )

        # Inter branch: long-token
        inter_feat = 0
        if self.enable_inter:
            # 每个空间位置的 token = [u1||u2||u3||u4]，维度 4*C0
            u_cat = torch.cat([u1, u2, u3, u4], dim=1)  # (B,4*C0,D',H',W')
            inter_cat = self.mamba_inter(u_cat)         # (B,4*C0,D',H',W')
            inter_feat = self.inter_reduce(inter_cat)   # (B,C0,D',H',W')

        # fuse
        y = 0
        if self.enable_intra:
            y = y + intra_sum
        if self.enable_inter:
            y = y + inter_feat

        if self.add_residual_from_embed:
            y = y + embed_sum

        # upsample to full resolution
        y = self.up_sample(y)
        return y
