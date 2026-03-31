import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
import torch._dynamo as dynamo
from hilbertcurve.hilbertcurve import HilbertCurve
from functools import lru_cache


# =========================================================================
#  Helper Functions
# =========================================================================

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _ilog2_pow2(n: int) -> int:
    return int(math.log2(n))


def _hilbert_i2c(n: int, p: int, h: int):
    """Convert Hilbert distance 'h' to n-dimensional coordinates."""
    x = [0] * n
    for i in range(p):
        for j in range(n):
            bit = (h >> (i * n + (n - 1 - j))) & 1
            x[j] |= bit << i
    t = x[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        x[i] ^= x[i - 1]
    x[0] ^= t
    Q = 2
    while Q < (1 << p):
        P = Q - 1
        for i in range(n - 1, -1, -1):
            if x[i] & Q:
                x[0] ^= P
            else:
                t = (x[0] ^ x[i]) & P
                x[0] ^= t
                x[i] ^= t
        Q <<= 1
    return x


@dynamo.disable
def _get_fine_hilbert_idx(N, device):
    N = int(N)
    idx_f = _hilbert_idx_3d_linear(N).to(device)
    inv_f = _invert_permutation(idx_f)
    return idx_f, inv_f


@dynamo.disable
def _get_coarse_block_target_idx(N, target_blocks, device):
    N = int(N)
    target_blocks = int(target_blocks)
    idx_c = _block_hilbert_interleave_idx_target(N, target_blocks).to(device)
    inv_c = _invert_permutation(idx_c)
    return idx_c, inv_c


@lru_cache(maxsize=None)
def _hilbert_idx_2d_linear(N: int) -> torch.Tensor:
    if not _is_pow2(N):
        raise ValueError(f"2D Hilbert requires N be power-of-two, got N={N}")
    p = _ilog2_pow2(N)
    L = N * N
    hc = HilbertCurve(p, 2)
    idx = [0] * L
    for t in range(L):
        x, y = hc.point_from_distance(t)
        idx[t] = y * N + x
    return torch.tensor(idx, dtype=torch.long)


@lru_cache(maxsize=None)
def _hilbert_idx_3d_linear(N: int) -> torch.Tensor:
    if not _is_pow2(N):
        raise ValueError(f"3D Hilbert requires N be power-of-two, got N={N}")
    p = _ilog2_pow2(N)
    L = N * N * N
    hc = HilbertCurve(p, 3)
    idx = [0] * L
    for t in range(L):
        x, y, z = hc.point_from_distance(t)
        d, h, w = z, y, x
        idx[t] = d * (N * N) + h * N + w
    return torch.tensor(idx, dtype=torch.long)


@lru_cache(maxsize=None)
def _slice_hilbert_interleave_idx(N: int) -> torch.Tensor:
    """[Optimized] Vectorized Slice-Hilbert-Interleave."""
    if not _is_pow2(N):
        raise ValueError(f"Slice-Hilbert-Interleave requires N be power-of-two, got N={N}")
    plane = _hilbert_idx_2d_linear(N)
    base_stride = N * N
    depth_offsets = torch.arange(N, dtype=plane.dtype, device=plane.device) * base_stride
    interleaved = plane.unsqueeze(1) + depth_offsets.unsqueeze(0)
    return interleaved.reshape(-1)


def _choose_block_grid_dims(target_blocks: int, N: int) -> tuple[int, int, int]:
    if not _is_pow2(target_blocks):
        raise ValueError(f"target_blocks must be power-of-two, got {target_blocks}")
    if not _is_pow2(N):
        raise ValueError(f"N must be power-of-two, got {N}")
    if target_blocks <= 0 or target_blocks > N ** 3:
        raise ValueError(f"Invalid target_blocks={target_blocks}")

    k = _ilog2_pow2(target_blocks)
    best = None
    best_score = float("inf")
    for a in range(k + 1):
        for b in range(k + 1 - a):
            c = k - a - b
            gd, gh, gw = (1 << a), (1 << b), (1 << c)
            if gd > N or gh > N or gw > N: continue
            bd, bh, bw = N // gd, N // gh, N // gw
            score = max(bd, bh, bw) / max(1, min(bd, bh, bw))
            if score < best_score:
                best_score = score
                best = (gd, gh, gw)
    if best is None:
        raise ValueError(f"Cannot factor target_blocks={target_blocks}")
    return best


@lru_cache(maxsize=None)
def _block_hilbert_interleave_idx_target(N: int, target_blocks: int) -> torch.Tensor:
    """[Optimized] Vectorized Block-Hilbert-Interleave."""
    if not _is_pow2(N):
        raise ValueError(f"N must be power-of-two, got N={N}")

    gd, gh, gw = _choose_block_grid_dims(target_blocks, N)
    bd, bh, bw = N // gd, N // gh, N // gw
    S = target_blocks

    gmax = max(gd, gh, gw)
    p = _ilog2_pow2(gmax)
    hc = HilbertCurve(p, 3)

    block_starts = []
    block_stride_d = bd * N * N
    block_stride_h = bh * N
    block_stride_w = bw

    count = 0
    # Python loop is fine here (S is small)
    for t in range(gmax ** 3):
        x, y, z = hc.point_from_distance(t)
        if (x < gw) and (y < gh) and (z < gd):
            start_idx = z * block_stride_d + y * block_stride_h + x * block_stride_w
            block_starts.append(start_idx)
            count += 1
            if count == S: break

    block_starts_tensor = torch.tensor(block_starts, dtype=torch.long)

    range_d = torch.arange(bd) * (N * N)
    range_h = torch.arange(bh) * N
    range_w = torch.arange(bw)

    # Vectorized broadcasting
    intra_offsets = range_d.view(-1, 1, 1) + range_h.view(1, -1, 1) + range_w.view(1, 1, -1)
    intra_offsets_flat = intra_offsets.flatten()

    final_indices = intra_offsets_flat.unsqueeze(1) + block_starts_tensor.unsqueeze(0)
    return final_indices.flatten()


def _invert_permutation(idx: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), dtype=idx.dtype, device=idx.device)
    return inv


# =========================================================================
#  Mamba Ops Imports
# =========================================================================
try:
    from mamba_mims.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    mamba_inner_fn_no_out_proj = None

try:
    from mamba_mims.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_mims.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# =========================================================================
#  Main Mamba Class
# =========================================================================

class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            nslices_small=64,
            coarse_mode: str = "block_hilbert_interleave",
            block_grid: Optional[int] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.nslices_small = nslices_small
        self.coarse_mode = coarse_mode
        self.block_grid = block_grid  # Still keeping parameter for compatibility but logic is simplified

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # B-direction params
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True

        # Spatial params
        A_s = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_log = torch.log(A_s)
        self.A_s_log = nn.Parameter(A_s_log)
        self.A_s_log._no_weight_decay = True

        self.conv1d_s = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.D_s = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_s._no_weight_decay = True

        # Reverse spatial params
        A_s_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_b_log = torch.log(A_s_b)
        self.A_s_b_log = nn.Parameter(A_s_b_log)
        self.A_s_b_log._no_weight_decay = True

        self.conv1d_s_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.D_s_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_s_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        # Common Projection
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        # Determine N and Indexes
        N = self.nslices_small
        if N is None:
            N = int(round(seqlen ** (1.0 / 3.0)))
        use_hilbert = (N * N * N == seqlen)

        if use_hilbert:
            # 1. Fine Indices
            idx_f, inv_f = _get_fine_hilbert_idx(N, xz.device)

            # 2. Coarse Indices (Simplified Logic)
            if self.coarse_mode == "slice_hilbert_interleave":
                idx_c = _slice_hilbert_interleave_idx(N).to(xz.device)
                inv_c = _invert_permutation(idx_c)
            elif self.coarse_mode == "block_hilbert_interleave":
                # Always use the optimized target_idx function.
                # If block_grid was provided (legacy), use it cubed. Else use nslices_small.
                target_blocks = self.nslices_small if self.nslices_small is not None else N
                if self.block_grid is not None:
                    target_blocks = self.block_grid ** 3

                idx_c, inv_c = _get_coarse_block_target_idx(N, target_blocks, xz.device)
            else:
                idx_c, inv_c = None, None
        else:
            idx_f = inv_f = idx_c = inv_c = None

        A_b = -torch.exp(self.A_b_log.float())

        # === Fine Branch (3D Hilbert) ===
        if idx_f is not None:
            xz_f = xz.index_select(-1, idx_f)
        else:
            xz_f = xz

        out_f = mamba_inner_fn_no_out_proj(
            xz_f, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight,
            A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )
        out_f_b = mamba_inner_fn_no_out_proj(
            xz_f.flip([-1]), self.conv1d_b.weight, self.conv1d_b.bias, self.x_proj_b.weight, self.dt_proj_b.weight,
            A_b, None, None, self.D_b.float(), delta_bias=self.dt_proj_b.bias.float(), delta_softplus=True,
        )
        out_f = out_f + out_f_b.flip([-1])
        if inv_f is not None:
            out_f = out_f.index_select(-1, inv_f)

        # === Coarse Branch ===
        A_s = -torch.exp(self.A_s_log.float())
        A_s_b = -torch.exp(self.A_s_b_log.float())

        if idx_c is not None:
            xz_s = xz.index_select(-1, idx_c)
        else:
            # Fallback
            xz_s = xz.chunk(self.nslices_small, dim=-1)
            xz_s = torch.stack(xz_s, dim=-1)
            xz_s = xz_s.flatten(-2)

        out_s = mamba_inner_fn_no_out_proj(
            xz_s, self.conv1d_s.weight, self.conv1d_s.bias, self.x_proj_s.weight, self.dt_proj_s.weight,
            A_s, None, None, self.D_s.float(), delta_bias=self.dt_proj_s.bias.float(), delta_softplus=True,
        )
        xz_s_b = xz_s.flip([-1])
        out_s_b = mamba_inner_fn_no_out_proj(
            xz_s_b, self.conv1d_s_b.weight, self.conv1d_s_b.bias, self.x_proj_s_b.weight, self.dt_proj_s_b.weight,
            A_s_b, None, None, self.D_s_b.float(), delta_bias=self.dt_proj_s_b.bias.float(), delta_softplus=True,
        )

        out_c = out_s + out_s_b.flip([-1])
        if inv_c is not None:
            out_c = out_c.index_select(-1, inv_c)
        else:
            out_c = out_c.reshape(batch, self.d_inner, seqlen // self.nslices_small, self.nslices_small) \
                .permute(0, 1, 3, 2).flatten(-2)

        out_sq = F.linear(rearrange(out_f, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        out_slice = F.linear(rearrange(out_c, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

        return out_sq, out_slice

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv,
                device=self.conv1d.weight.device, dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_state,
                device=self.dt_proj.weight.device, dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        # ... (Step logic kept same if needed for inference, though rarely used in training)
        dtype = hidden_states.dtype
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None: x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state