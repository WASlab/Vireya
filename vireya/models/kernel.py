#Borrowed Deepseek-V3's kernel.py verbatim
# kernel.py (fallback-aware version)
from typing import Tuple
import torch
import os

# Check for Triton availability and Ampere+ architecture
try:
    import triton
    import triton.language as tl
    from triton import Config
    CC_MAJOR = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    TRITON_AVAILABLE = CC_MAJOR >= 8  # Ampere or newer
except ImportError:
    triton = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.
        y = x / s
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    if TRITON_AVAILABLE:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
        return y, s
    else:
        scale = x.abs().amax(dim=-1, keepdim=True) / 448.
        y = (x / scale).to(torch.float8_e4m3fn)
        return y, scale.squeeze(-1)


if TRITON_AVAILABLE:
    @triton.jit
    def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        n = tl.cdiv(N, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * n + pid_n)
        y = x * s
        tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    if TRITON_AVAILABLE:
        y = torch.empty_like(x, dtype=torch.get_default_dtype())
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
        weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
        return y
    else:
        return (x.float() * s).to(torch.get_default_dtype())


if TRITON_AVAILABLE:
    fp8_gemm_configs = [
        Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=stg, num_warps=8)
        for block_m in [16, 32, 64] for block_n in [32, 64, 128] for stg in [3, 4, 5, 6]
    ]

    @triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
    @triton.jit
    def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                        a_s_ptr, b_s_ptr,
                        M, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        k = tl.cdiv(K, BLOCK_SIZE_K)
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
        a_s_ptrs = a_s_ptr + offs_m * k
        b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(k):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
            a_s = tl.load(a_s_ptrs)
            b_s = tl.load(b_s_ptrs)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
            a_s_ptrs += 1
            b_s_ptrs += 1

        c = acc.to(c_ptr.dtype.element_ty)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    if TRITON_AVAILABLE:
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
        fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
        return c
    else:
        a_deq = a.float() * a_s.unsqueeze(-1)
        b_deq = b.float() * b_s.unsqueeze(0)
        return torch.matmul(a_deq, b_deq.t())