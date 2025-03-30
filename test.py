import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import wandb
from torchvision import datasets, transforms

from vireya.models.model import Transformer
from vireya.models.configs import small_config, base_config, tiny_config, large_config
from vireya.utils.optim import Muon, StepLRScheduler


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def adjust_configuration(model):
    """
    Randomly adjusts configurations in case of NaNs during forward pass.
    """
    new_kv_rank = random.choice([16, 32, 64, 128])
    print(f"Adjusting KV rank to {new_kv_rank}")
    
    model.kv_lora_rank = new_kv_rank
    model.wkv_a = Linear(model.dim, model.kv_lora_rank + model.qk_rope_head_dim)
    model.kv_norm = DyT(model.kv_lora_rank)
    model.wkv_b = ColumnParallelLinear(model.kv_lora_rank, model.n_heads * (model.qk_nope_head_dim + model.v_head_dim))

    return {'kv_lora_rank': new_kv_rank}


def check_for_nans_and_retry(model, input_tensor, freqs_cis, max_retries=5):
    """
    Tries the forward pass and retries with new configurations if NaNs are encountered.
    """
    for attempt in range(max_retries):
        output = model(input_tensor, freqs_cis=freqs_cis)
        
        # Check for NaNs in the output
        if torch.isnan(output).any():
            print(f"NaNs detected on attempt {attempt + 1}. Retrying with modified configuration...")
            new_config = adjust_configuration(model)
            model.apply_configuration(new_config)
        else:
            print(f"Stable output obtained on attempt {attempt + 1}.")
            return output, True

    print(f"Failed to stabilize after {max_retries} attempts.")
    return None, False


def train(rank, world_size, args):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16

    config_map = {
        "tiny": tiny_config,
        "small": small_config,
        "base": base_config,
        "large": large_config
    }

    # Using predefined "small" model config for now, but this could be adjusted dynamically.
    model_args = small_config(dtype=args.dtype)
    model_args.max_batch_size = args.batch_size

    model = Transformer(model_args).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer and Scheduler
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'embedding' not in name and 'head' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizer = Muon(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=args.lr,
        wd=args.weight_decay,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
    )

    scheduler = StepLRScheduler(
        optimizer,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        init_lr=args.init_lr,
        peak_lr=args.lr,
        end_lr=args.end_lr,
        warmup_type="cosine",
        decay_type="cosine"
    )

    # CIFAR-10 dataset transformation and loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
        writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device, dtype=dtype, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Fault tolerant forward pass
            output, success = check_for_nans_and_retry(model, inputs, freqs_cis=None)
            if not success:
                print("Model configuration failed to stabilize.")
                continue

            loss = criterion(output, targets)
            print("Logits:", output.min().item(), output.max().item(), output.std().item())
            print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            if rank == 0:
                lr = scheduler.get_lr()
                wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", lr, global_step)

            global_step += 1

    if rank == 0:
        wandb.finish()
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--end_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tensorboard_logdir", type=str, default="runs")
    parser.add_argument("--project_name", type=str, default="efficient-vit")
    parser.add_argument("--dataset", type=str, default="cifar10")  # Always CIFAR-10
    parser.add_argument("--model", type=str, choices=["tiny", "small", "base", "large"], default="small")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp8"], default="bf16")

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size)
