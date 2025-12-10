# src/comparative/utils/hardware.py

import multiprocessing as mp
import os
import torch
from torch import distributed as dist
from datetime import timedelta
from rich import print as rprint

def device():
    """Best available torch.device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def num_devices():
    """Returns logical GPU count or 1 for CPU/MPS."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

def num_workers(max_cap=12):
    """Returns # workers for DataLoader (defaults to half logical cores, capped)."""
    env = os.getenv("NUM_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return min(max_cap, max(1, mp.cpu_count() // 2))

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def init_ddp(backend=None, timeout=1800):
    if is_distributed():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return
    chosen_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(
        backend=chosen_backend,
        timeout=timedelta(seconds=timeout)
    )
    rprint(
        f"[bold green]DDP initialized[/] (rank {dist.get_rank()}/{world_size}, backend {chosen_backend})"
    )

def cleanup_ddp():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
