import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

from vireya.models.model import Transformer
from vireya.models.configs import tiny_config, small_config, base_config, large_config
from data import get_dataset
from vireya.utils.optim import Muon, StepLRScheduler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def evaluate(model, val_loader, device, dtype):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, dtype=dtype, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    torch.set_default_dtype(dtype)

    config_map = {
        "tiny": tiny_config,
        "small": small_config,
        "base": base_config,
        "large": large_config
    }

    model_args = config_map[args.model](dtype=args.dtype)
    model_args.max_batch_size = args.batch_size
    model = Transformer(model_args).to(device)

    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

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

    train_dataset, val_dataset = get_dataset(name=args.dataset, data_dir=getattr(args, "data_dir", "./data"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True),
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
        writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device, dtype=dtype, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if rank == 0 and global_step % 10 == 0:
                lr = scheduler.get_lr()
                wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", lr, global_step)

            global_step += 1

        # Evaluate after each epoch
        if rank == 0:
            val_acc = evaluate(model, val_loader, device, dtype)
            wandb.log({"val_accuracy": val_acc, "epoch": epoch})
            writer.add_scalar("Accuracy/val", val_acc, epoch)

    if rank == 0:
        wandb.finish()
        writer.close()
    cleanup()

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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["tiny", "small", "base", "large"], default="small")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    mp.spawn(train, args=(torch.cuda.device_count(), args), nprocs=torch.cuda.device_count())
