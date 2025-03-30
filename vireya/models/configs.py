# vireya/models/configs.py

from vireya.models.model import ModelArgs

def tiny_config(dtype: str = "bf16") -> ModelArgs:
    seq_len = (32 // 4) ** 2
    return ModelArgs(
        image_size=32,
        patch_size=4,
        dim=128,
        inter_dim=512,
        moe_inter_dim=384,
        n_layers=6,
        n_dense_layers=1,
        n_heads=3,
        n_routed_experts=4,
        n_shared_experts=2,
        n_activated_experts=2,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        kv_lora_rank=32,
        v_head_dim=32,
        max_seq_len=seq_len + 1,
        original_seq_len=seq_len,
        dtype=dtype,
        
        
    )

def small_config(dtype: str = "bf16") -> ModelArgs:
    seq_len = (32 // 4) ** 2
    return ModelArgs(
        image_size=32,
        patch_size=4,
        dim=192,
        inter_dim=768,
        moe_inter_dim=512,
        n_layers=12,
        n_dense_layers=1,
        n_heads=6,
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        kv_lora_rank=16, 
        max_seq_len=seq_len + 1,
        original_seq_len=seq_len,
        dtype=dtype,
    )


def base_config(dtype: str = "bf16") -> ModelArgs:
    seq_len = (224 // 16) ** 2
    return ModelArgs(
        image_size=224,
        patch_size=16,
        dim=768,
        inter_dim=3072,
        moe_inter_dim=2048,
        n_layers=12,
        n_dense_layers=2,
        n_heads=12,
        n_routed_experts=16,
        n_shared_experts=2,
        n_activated_experts=2,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        max_seq_len=seq_len + 1,
        original_seq_len=seq_len,
        dtype=dtype,
    )

def large_config(dtype: str = "bf16") -> ModelArgs:
    seq_len = (224 // 16) ** 2
    return ModelArgs(
        image_size=224,
        patch_size=16,
        dim=1024,
        inter_dim=4096,
        moe_inter_dim=3072,
        n_layers=24,
        n_dense_layers=2,
        n_heads=16,
        n_routed_experts=32,
        n_shared_experts=2,
        n_activated_experts=4,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        max_seq_len=seq_len + 1,
        original_seq_len=seq_len,
        dtype=dtype,
    )
