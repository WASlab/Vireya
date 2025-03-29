# model_stats.py

import torch
from model import ModelArgs, Transformer 

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_active_parameters(model: Transformer, args: ModelArgs):
    """
    Estimates number of parameters *used* per forward pass:
    - Includes all dense layers
    - Includes only top-k experts per MoE layer
    """
    total = 0

    # Embedding
    total += sum(p.numel() for p in model.embed.parameters())

    for i, layer in enumerate(model.layers):
        # Attention: always fully active
        total += sum(p.numel() for p in layer.attn.parameters())

        # FFN: either MLP or MoE
        if isinstance(layer.ffn, torch.nn.Module):
            if isinstance(layer.ffn, torch.nn.Sequential):  # MLP fallback
                total += sum(p.numel() for p in layer.ffn.parameters())
            elif hasattr(layer.ffn, "experts"):  # MoE
                moe = layer.ffn

                # Shared MLPs: always active
                total += sum(p.numel() for p in moe.shared_experts.parameters())

                # Active experts: top-k out of n
                k = args.n_activated_experts
                expert_params = sum(p.numel() for p in moe.experts[0].parameters())
                total += k * expert_params
        else:
            total += sum(p.numel() for p in layer.ffn.parameters())

        # Norms
        total += sum(p.numel() for p in layer.attn_norm.parameters())
        total += sum(p.numel() for p in layer.ffn_norm.parameters())

    # Final norm and head
    total += sum(p.numel() for p in model.norm.parameters())
    total += sum(p.numel() for p in model.head.parameters())
    return total

def readable(n):
    return f"{n/1e6:.2f}M"

def run(config_name: str):
    def get_config(size):
        if size == "S":
            return ModelArgs(dim=384, inter_dim=1536, moe_inter_dim=1024, n_layers=8, n_heads=6)
        elif size == "M":
            return ModelArgs(dim=512, inter_dim=2048, moe_inter_dim=1536, n_layers=12, n_heads=8)
        elif size == "L":
            return ModelArgs(dim=768, inter_dim=3072, moe_inter_dim=2048, n_layers=24, n_heads=12)
        raise ValueError(f"Unknown config: {size}")
    
    args = get_config(config_name)
    model = Transformer(args)

    total, trainable = count_parameters(model)
    active = estimate_active_parameters(model, args)

    print(f"\n[Model Configuration: {config_name}]")
    print(f"Total Parameters:    {readable(total)}")
    print(f"Trainable Parameters:{readable(trainable)}")
    print(f"Active Parameters:   {readable(active)}  (per forward pass)")
    print(f"Sparsity Efficiency: {100 * active / total:.2f}%")

if __name__ == "__main__":
    for size in ["S", "M", "L"]:
        run(size)
