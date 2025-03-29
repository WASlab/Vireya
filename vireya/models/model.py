import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,Optional,Literal
from dataclasses import dataclass
from torch import Tensor
import math
from vireya.models.kernel import act_quant, weight_dequant, fp8_gemm
import torch.distributed as dist

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters in a Vision + MoE context.
    
    Attributes:
        image_size (int): Resolution of the input images (e.g. 224 for ImageNet).
        patch_size (int): Spatial patch size (e.g. 16).
        num_classes (int): Number of output classes for classification tasks.
        dim (int): Model dimension (hidden size).
        inter_dim (int): Intermediate dimension for dense MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE experts.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): How many of those layers are standard MLP layers before MoE.
        n_heads (int): Number of attention heads.
        
        # MoE parameters
        n_routed_experts (int): Total number of routed experts.
        n_shared_experts (int): Number of shared experts (as in shared MLP).
        n_activated_experts (int): How many experts are chosen for each token.
        n_expert_groups (int): Number of groups for group-wise routing.
        n_limited_groups (int): Number of those groups that can route tokens (top-k gating).
        score_func (Literal['softmax','sigmoid']): Scoring function for gating.
        route_scale (float): Additional scaling factor for gating scores.
        
        # MLA + RoPE parameters
        q_lora_rank (int): Low-rank dimension for Q projection if used (0 = off).
        kv_lora_rank (int): Low-rank dimension for KV projection if used (0 = off).
        qk_nope_head_dim (int): Dim per head for non-positional Q/K.
        qk_rope_head_dim (int): Dim per head for rotary-positional Q/K.
        v_head_dim (int): Dim per head for V.
        max_seq_len (int): Max number of patches (spatial tokens). Usually (image_size//patch_size)**2.
        original_seq_len (int): Base sequence length for computing rope corrections.
        rope_theta (float): Base for RoPE exponent.
        rope_factor (float): Additional factor for extended rope range.
        beta_fast (int): Fast beta for partial RoPE corrections.
        beta_slow (int): Slow beta for partial RoPE corrections.
        mscale (float): Additional multiplier if we scale attention for longer context.
        
        # Data types and distributed
        dtype (Literal['bf16','fp8']): Computation precision.
        world_size (int): Number of distributed processes (for parallel linear).
        max_batch_size (int): Max batch size for memory caching, etc.
    """
    # Vision
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000

    # Transformer dims
    dim: int = 768
    inter_dim: int = 3072
    moe_inter_dim: int = 2048
    n_layers: int = 12
    n_dense_layers: int = 2
    n_heads: int = 12

    # MoE
    n_routed_experts: int = 16
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0

    # MLA + RoPE
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    max_seq_len: int = (224 // 16) ** 2  # 14x14 = 196 by default
    original_seq_len: int = 196
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    world_size = 1
    rank = 0
    block_size = 128
    dtype: Literal["bf16", "fp8"] = "bf16"
    attn_impl: Literal["naive", "absorb"] = "absorb"
    
   
    max_batch_size = 1
class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for transforming image patches into a sequence of embeddings.

    Converts an input image of shape (B, 3, H, W) into a sequence of tokens (B, N, dim),
    where N = (H / patch_size) * (W / patch_size), plus one if cls_token is used.

    Attributes:
        patch_size (int): Size of the patches to be extracted from the input image.
        dim (int): Dimensionality of the output embeddings.
        projection (nn.Conv2d): Convolutional layer for patch extraction and embedding.
        cls_token (nn.Parameter): Optional class token prepended to patch embeddings.
    """
    def __init__(self, args: ModelArgs, use_cls_token: bool = False):
        """
        Initializes the PatchEmbedding layer.

        Args:
            args (ModelArgs): Model arguments containing patch size and embedding dimension.
            use_cls_token (bool): Whether to include a learnable class token.
        """
        super().__init__()
        self.patch_size = args.patch_size
        self.dim = args.dim
        self.image_size = args.image_size
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.use_cls_token = use_cls_token

        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size."

        self.projection = nn.Conv2d(
        in_channels=3,
        out_channels=self.dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        bias=True,
        dtype=Linear.dtype  # Match model precision (bf16 or fp8)
    )


        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        else:
            self.register_parameter("cls_token", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PatchEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, dim), where N = num_patches (+1 if cls_token used).
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Input image must be of shape ({self.image_size}, {self.image_size}), "
            f"but got ({H}, {W})"
        )

        x = self.projection(x)               # (B, dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)     # (B, N_patches, dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
            x = torch.cat((cls_tokens, x), dim=1)          # (B, N_patches+1, dim)

        return x

world_size: int = 1
rank: int = 0
block_size: int = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

class DyT(nn.Module):
    def __init__(self,dim,init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*init_alpha) #Learnable scalar (shared across dim)
        self.gamma = nn.Parameter(torch.ones(dim))          #Learnable per-channel scale
        self.beta = nn.Parameter(torch.zeros(dim))          #Learnable per-channel shift
        
        
    def forward(self,x):
        """

        Args:
            x: Input Tensor of Shape [B,T,C] or [*,C] for generic transformer input.

        Returns:
            Tensor of same shape as input after applying DyT
        """
        x = torch.tanh(self.alpha*x)
        
        return self.gamma*x + self.beta


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=self.weight.dtype))

        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y
    
class MLA(nn.Module):
    """
    Multi-Headed Latent Attention (MLA) Layer adapted for vision transformers using RoPE-Mixed.

    Supports LoRA-style KV compression and optional rotary embeddings on a per-head subset of dimensions.

    Args:
        args (ModelArgs): Configuration for the model.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size

        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank

        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.v_head_dim = args.v_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        # Query projection
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = DyT(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # Key/Value low-rank pre-projection and decomposition
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = DyT(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # Cache buffers for inference
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_nope_head_dim), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for MLA.

        Args:
            x (Tensor): Input tensor of shape (B, T, dim).
            start_pos (int): Starting position in the sequence.
            freqs_cis (Tensor): RoPE frequency tensor for rotary embedding.
            mask (Optional[Tensor]): Attention mask (causal or padding).

        Returns:
            Tensor: Output tensor of shape (B, T, dim).
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # Key/value compression
        kv = self.wkv_a(x)
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)  # shape: (B, T, 1, D) -> expand later

        if attn_impl == "naive":
            # Project k/v fully
            kv_proj = self.wkv_b(self.kv_norm(kv))
            kv_proj = kv_proj.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_full = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            self.k_cache[:bsz, start_pos:end_pos] = k_full
            self.v_cache[:bsz, start_pos:end_pos] = v

            q_full = torch.cat([q_nope, q_rope], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q_full, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            attn_out = torch.einsum("bsht,bthd->bshd", scores.softmax(dim=-1), self.v_cache[:bsz, :end_pos])

        else:
            # Compressed cache path
            kv_proj = self.kv_norm(kv)  # (B, T, kv_lora_rank)
            kv_proj = self.wkv_b(kv_proj)
            kv_proj = kv_proj.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            self.kv_cache[:bsz, start_pos:end_pos] = k_nope.detach()
            self.pe_cache[:bsz, start_pos:end_pos] = k_rope.expand(-1, -1, self.n_local_heads, -1).detach()

            scores = (
            torch.einsum("bshc,bthc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
            torch.einsum("bshc,bthc->bsht", q_rope, self.pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale

            attn_out = torch.einsum("bsht,bthc->bshc", scores.softmax(dim=-1), self.kv_cache[:bsz, :end_pos])
            wkv_b_weight = self.wkv_b.weight  # [kv_lora_rank, n_heads * (qk_nope + v)]
            if self.wkv_b.scale is not None:
                wkv_b_weight = weight_dequant(wkv_b_weight, self.wkv_b.scale, block_size)

            # Reshape to: [n_heads, qk_nope + v, kv_lora_rank]
            wkv_b_weight = wkv_b_weight.t().contiguous().view(
                self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )

            # Extract value projection weights: [n_heads, v_head_dim, kv_lora_rank]
            v_proj_w = wkv_b_weight[:, -self.v_head_dim:, :]  # this is v_proj

            # Transpose for matmul shape: [n_heads, kv_lora_rank, v_head_dim]
            v_proj_w = v_proj_w.transpose(-2, -1).contiguous()  # now [n_heads, kv_lora_rank, v_head_dim]

            # attn_out: [B, S, n_heads, kv_lora_rank]
            # v_proj_w: [n_heads, kv_lora_rank, v_head_dim]
            attn_out = torch.einsum("bshc,hcd->bshd", attn_out, v_proj_w)


        x = self.wo(attn_out.flatten(2))
        return x


    
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    bsz, seqlen, n_heads, dim = x.shape
    assert dim % 2 == 0

    # x: [bsz, seqlen, n_heads, dim//2, 2] → complex
    x_complex = torch.view_as_complex(x.float().reshape(bsz, seqlen, n_heads, dim//2, 2))

    # Always slice (do NOT reshape freqs_cis here!)
    freqs = freqs_cis[:seqlen].unsqueeze(0).unsqueeze(2)  # [1, seqlen, 1, dim//2] 

    out = x_complex * freqs

    out = torch.view_as_real(out).type(dtype)
    return out.reshape(bsz, seqlen, n_heads, dim)



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = DyT(args.dim)
        self.ffn_norm = DyT(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer model for vision tasks with support for RoPE-based attention, Mixture-of-Experts (MoE),
    and modular attention/feedforward layers. The architecture uses dynamic rotary embedding handling,
    supports distributed training, and includes a learnable patch embedding layer.

    Attributes:
        args (ModelArgs): Configuration object specifying model parameters.
        embed (PatchEmbedding): Converts input images into patch tokens with optional cls_token.
        layers (nn.ModuleList): A sequence of Transformer blocks (attention + MLP or MoE).
        norm (DyT): Final normalization layer before the classifier head.
        head (nn.Linear): Linear layer mapping the final representation to output logits.
        freqs_cis (Tensor): Buffer containing precomputed complex rotary embedding values.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Set the default dtype globally for all Linear layers
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.args = args

        # Image → patch embedding with optional class token
        self.embed = PatchEmbedding(args, use_cls_token=True)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            Block(i, args) for i in range(args.n_layers)
        ])

        # Normalization before classification head
        self.norm = DyT(args.dim)

        # Final projection to num_classes
        self.head = Linear(args.dim, args.num_classes,bias=True)

        # Precompute rotary embeddings up to max_seq_len
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of shape (B, C, H, W) representing image batches.
            start_pos (int, optional): Start position for positional encodings. Defaults to 0.

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        
        # tokens should be [B, T, dim] after PatchEmbedding

        # Patch embedding + cls_token (if enabled)
        h = self.embed(tokens)
        seqlen = h.size(1)
        # Slice the correct amount of rotary embeddings for the current sequence length
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # Causal attention mask if sequence length > 1
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)

        # Forward through transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # Final norm and select the last token (usually cls_token)
        h = self.norm(h)[:, -1]  # shape: (B, dim)

        # Project to class logits
        logits = self.head(h)

        # Optional distributed gather across devices
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        return logits


