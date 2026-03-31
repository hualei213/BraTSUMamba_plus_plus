import torch
import torch.nn as nn
from einops import rearrange
from mamba_AdM.BGM import Mamba_mims


# ==========================================
# 辅助类
# ==========================================

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
        return self.net(x)


class MambaLayer(nn.Module):
    """
    全序列尺度 Mamba 层，包含三个方向的扫描
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # 使用 BGM.py 中的 Mamba_mims
        self.mamba = Mamba_mims(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="Bi"
        )

    def mamba_forward(self, x):
        B, C = x.shape[:2]
        # x shape: (B, C, D, H, W)
        img_dims = x.shape[2:]
        n_tokens = img_dims.numel()

        # Flatten: (B, C, L) -> (B, L, C)
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x = self.norm(x)
        x = self.mamba(x)

        # Reshape back: (B, L, C) -> (B, C, L) -> (B, C, D, H, W)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)
        return x

    def forward(self, x):
        # 简单起见，这里实现三向特征融合

        # 1. 原始方向 (D, H, W)
        out_1 = self.mamba_forward(x)

        # 2. W-D 交换
        x_2 = rearrange(x, "b c d w h -> b c w d h")
        out_2 = self.mamba_forward(x_2)
        out_2 = rearrange(out_2, "b c w d h -> b c d w h")

        # 3. H-D 交换
        x_3 = rearrange(x, "b c d w h -> b c h w d")
        out_3 = self.mamba_forward(x_3)
        out_3 = rearrange(out_3, "b c h w d -> b c d w h")

        # 融合 + 原始残差
        return out_1 + out_2 + out_3 + x

    # ==========================================


# MIMS 主类 (带上采样)
# ==========================================

class MIMS(nn.Module):
    def __init__(
            self,
            out_dim: int = 32,  # 目标输出通道 (对应 Encoder Stage 0 的输出)
            c0: int = 16,  # MIMS 内部计算通道
            embed_type: str = "conv",
            embed_stride: int = 2,  # 内部下采样: 128 -> 64
            directions=("w", "h", "d"),
            mamba_kwargs: dict | None = None,
            add_residual_from_embed: bool = True,
    ):
        super().__init__()

        self.embed_stride = embed_stride

        # 1. 独立 Embed (下采样至 64^3)
        self.embeds = nn.ModuleList(
            [PerModalityEmbed(c0=c0, embed_type=embed_type, stride=embed_stride) for _ in range(4)]
        )

        # 2. 共享权重的 Mamba 层 (在 64^3 上运行)
        if mamba_kwargs is None:
            mamba_kwargs = dict(d_state=16, d_conv=4, expand=2)

        self.mamba_layer = MambaLayer(
            dim=c0,
            d_state=mamba_kwargs.get('d_state', 16),
            d_conv=mamba_kwargs.get('d_conv', 4),
            expand=mamba_kwargs.get('expand', 2)
        )

        self.add_residual_from_embed = add_residual_from_embed

        # 3. [关键修改] 上采样恢复 (64^3 -> 128^3)
        # 使用 ConvTranspose3d 将分辨率变回 128，同时通道数变到 32
        if self.embed_stride > 1:
            self.up_sample = nn.ConvTranspose3d(
                in_channels=c0,
                out_channels=out_dim,
                kernel_size=embed_stride,
                stride=embed_stride,
                bias=False
            )
        else:
            # 如果没下采样，用 1x1 卷积调整通道
            self.up_sample = nn.Conv3d(c0, out_dim, kernel_size=1, bias=False)

        # 初始化上采样层
        nn.init.kaiming_normal_(self.up_sample.weight, a=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, 4, 128, 128, 128)

        x_t1ce = x[:, 0:1, :, :, :]
        x_t1n = x[:, 1:2, :, :, :]
        x_t2f = x[:, 2:3, :, :, :]
        x_t2w = x[:, 3:4, :, :, :]

        # Embed (-> 64^3)
        u_t1ce = self.embeds[0](x_t1ce)
        u_t1n = self.embeds[1](x_t1n)
        u_t2f = self.embeds[2](x_t2f)
        u_t2w = self.embeds[3](x_t2w)

        # Mamba (-> 64^3)
        out_t1ce = self.mamba_layer(u_t1ce)
        out_t1n = self.mamba_layer(u_t1n)
        out_t2f = self.mamba_layer(u_t2f)
        out_t2w = self.mamba_layer(u_t2w)

        # 融合
        y = out_t1ce + out_t1n + out_t2f + out_t2w
        if self.add_residual_from_embed:
            y = y + (u_t1ce + u_t1n + u_t2f + u_t2w)

        # 上采样恢复 (-> 128^3, Channel=32)
        # 输出形状: (B, 32, 128, 128, 128)
        y = self.up_sample(y)

        return y