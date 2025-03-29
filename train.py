import os
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

from vireya.models.model import Transformer
from vireya.models.configs import small_config, base_config, tiny_config, large_config
from data import get_dataset
from vireya.utils.optim import Muon, StepLRScheduler

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

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

    model_args = config_map[args.model](dtype=args.dtype)
    
    
    model_args.max_batch_size = args.batch_size

    model = Transformer(model_args).to(device)
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
    lr=args.lr,  # for Muon
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

    train_dataset, val_dataset = get_dataset(
    name=args.dataset,
    data_dir=args.data_dir if hasattr(args, "data_dir") else "./data"
)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

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
            logits = model(inputs)
            loss = criterion(logits, targets)
            

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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["tiny", "small", "base", "large"], default="small")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp8"], default="bf16")

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size)
