import torch
import torch.nn as nn
from mamba_AdM.BGM import Mamba_mims
# Optional: if you want to directly use a custom Mamba_mims implementation without providing




class PerModalityEmbed(nn.Module):
    """Per-modality embedding for 3D MRI.

    Args:
        c0: output channels per modality.
        embed_type: 'conv' or 'dsconv'.
        stride: 1/2/4.

    Input:
        x: (B, 1, D, H, W)
    Output:
        (B, c0, D', H', W')
    """

    def __init__(self, c0: int, embed_type: str = "conv", stride: int = 2):
        super().__init__()
        assert embed_type in ("conv", "dsconv")
        assert stride in (1, 2, 4)

        layers = [
            nn.Conv3d(1, c0, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.InstanceNorm3d(c0),
            nn.GELU(),
        ]

        if embed_type == "dsconv":
            # depthwise (after expansion) + pointwise
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


def _interleave(U: torch.Tensor, axis: str):
    """
    U: (B, M, C, D, H, W)
    Optimized: 将非扫描轴放入 Batch 维度，极大降低序列长度 L
    Return:
      seq: (New_Batch, L_short, C)
      meta: (B, M, C, D, H, W, axis)
    """
    assert axis in ("w", "h", "d")
    B, M, C, D, H, W = U.shape

    if axis == "w":
        # 扫描 W 轴
        # 原始维度: (B, M, C, D, H, W)
        # 目标: 把 D, H 移到 Batch (dim 0), 把 W, M 留在 Sequence (dim 1)
        # Permute: (B, D, H, W, M, C)
        t = U.permute(0, 3, 4, 5, 1, 2).contiguous()
        # View: Batch=(B*D*H), Seq=(W*M), Dim=C
        seq = t.view(B * D * H, W * M, C)

    elif axis == "h":
        # 扫描 H 轴
        # Permute: (B, D, W, H, M, C)
        t = U.permute(0, 3, 5, 4, 1, 2).contiguous()
        # View: Batch=(B*D*W), Seq=(H*M), Dim=C
        seq = t.view(B * D * W, H * M, C)

    else:  # "d"
        # 扫描 D 轴
        # Permute: (B, H, W, D, M, C)
        t = U.permute(0, 4, 5, 3, 1, 2).contiguous()
        # View: Batch=(B*H*W), Seq=(D*M), Dim=C
        seq = t.view(B * H * W, D * M, C)

    meta = (B, M, C, D, H, W, axis)
    return seq, meta


def _deinterleave(out: torch.Tensor, meta):
    """
    out: (New_Batch, L_short, C)
    return: (B, M, C, D, H, W)
    """
    B, M, C, D, H, W, axis = meta

    if axis == "w":
        # out: (B*D*H, W*M, C) -> view (B, D, H, W, M, C)
        t = out.view(B, D, H, W, M, C)
        # permute back to (B, M, C, D, H, W)
        # Indices: 0(B), 4(M), 5(C), 1(D), 2(H), 3(W)
        U_out = t.permute(0, 4, 5, 1, 2, 3).contiguous()

    elif axis == "h":
        # out: (B*D*W, H*M, C) -> view (B, D, W, H, M, C)
        t = out.view(B, D, W, H, M, C)
        # permute back to (B, M, C, D, H, W)
        # Need: 0, 4, 5, 1(D), 3(H), 2(W)
        U_out = t.permute(0, 4, 5, 1, 3, 2).contiguous()

    else:  # "d"
        # out: (B*H*W, D*M, C) -> view (B, H, W, D, M, C)
        t = out.view(B, H, W, D, M, C)
        # permute back to (B, M, C, D, H, W)
        # Need: 0, 4, 5, 3(D), 1(H), 2(W)
        U_out = t.permute(0, 4, 5, 3, 1, 2).contiguous()

    return U_out



class MIMS(nn.Module):
    """MIMS (Modality-Interleaved Multi-directional Scan) stem.

    This module:
      1) embeds each modality separately (Conv or Conv+DSConv),
      2) performs interleaved scanning along selected axes (w/h/d),
      3) uses ONLY out_wq (2nd output from your Mamba: out_slice),
      4) fuses 4 modalities into a single stream feature map.

    Expected input:
      x: (B, 4, D, H, W)
    Output:
      y: (B, out_dim, D', H', W') where D' depends on embed_stride.
    """

    def __init__(
        self,
        out_dim: int = 16,
        c0: int = 16,
        embed_type: str = "conv",
        embed_stride: int = 2,
        directions=("w", "h", "d"),
        mamba_cls=None,
        mamba_kwargs: dict | None = None,
        share_mamba_across_dirs: bool = True,
        use_wq_only: bool = True,
        add_residual_from_embed: bool = True,
    ):
        super().__init__()
        assert directions, "directions cannot be empty"
        for d in directions:
            assert d in ("w", "h", "d")
        self.directions = directions
        self.use_wq_only = use_wq_only
        self.add_residual_from_embed = add_residual_from_embed
        # modality-specific embeddings

        self.embeds = nn.ModuleList(
            [PerModalityEmbed(c0=c0, embed_type=embed_type, stride=embed_stride) for _ in range(4)]
        )

        # scanners (direct build; no mamba_factory_REMOVED)
        if mamba_cls is None:
            mamba_cls = Mamba_mims
        if mamba_kwargs is None:
            mamba_kwargs = {}

        def _build_mamba():
            return mamba_cls(**mamba_kwargs)

        if share_mamba_across_dirs:
            m = _build_mamba()
            self.mamba_w = m
            self.mamba_h = m
            self.mamba_d = m
        else:
            self.mamba_w = _build_mamba()
            self.mamba_h = _build_mamba()
            self.mamba_d = _build_mamba()

        # fuse modalities into out_dim
        self.fuse = nn.Sequential(
            nn.Conv3d(4 * c0, out_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_dim),
        )
        self.res_fuse = nn.Conv3d(4 * c0, out_dim, kernel_size=1, bias=False)

    def _scan_axis(self, U: torch.Tensor, axis: str) -> torch.Tensor:
        """U: (B,4,c0,D',H',W') -> returns same shape after scan on chosen axis."""
        seq, meta = _interleave(U, axis)  # (B, L, c0)

        if axis == "w":
            mamba = self.mamba_w
        elif axis == "h":
            mamba = self.mamba_h
        else:
            mamba = self.mamba_d
        p = next(mamba.parameters())
        if seq.device != p.device or seq.dtype != p.dtype:
            seq = seq.to(device=p.device, dtype=p.dtype)

        out = mamba(seq)

        # Compatible with:
        #   - mamba(seq) -> out
        #   - mamba(seq) -> (out_sq, out_wq)
        if isinstance(out, (tuple, list)):
            out_sq, out_wq = out
            out = out_wq if self.use_wq_only else out_sq
        # else: treat `out` as out_wq by default

        U_out = _deinterleave(out, meta)
        return U_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5 and x.size(1) == 4, "Expected input shape (B,4,D,H,W)"

        # per-modality embed
        Us = []
        for m in range(4):
            Um = self.embeds[m](x[:, m : m + 1])  # (B,c0,D',H',W')
            Us.append(Um)
        U = torch.stack(Us, dim=1)  # (B,4,c0,D',H',W')

        U_acc = U
        if "w" in self.directions:
            U_acc = U_acc + self._scan_axis(U, "w")
        if "h" in self.directions:
            U_acc = U_acc + self._scan_axis(U, "h")
        if "d" in self.directions:
            U_acc = U_acc + self._scan_axis(U, "d")

        B, M, C0, Dp, Hp, Wp = U_acc.shape
        U_cat = U_acc.reshape(B, M * C0, Dp, Hp, Wp)  # (B,4*c0,D',H',W')
        y = self.fuse(U_cat)

        if self.add_residual_from_embed:
            y = y + self.res_fuse(U.reshape(B, M * C0, Dp, Hp, Wp))

        return y

