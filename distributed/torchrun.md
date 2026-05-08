# torchrun: Launch Configuration

## Rule: ALWAYS Use torchrun
Never launch distributed training with bare `python`. torchrun handles:
- Process spawning and rank assignment
- `LOCAL_RANK`, `RANK`, `WORLD_SIZE` env vars
- Fault tolerance and elastic training

## Single Node Launch

```bash
# Basic: 4 GPUs on 1 node
torchrun --standalone --nproc-per-node=4 train.py

# With config file
torchrun --standalone --nproc-per-node=8 train.py --config configs/ddp_8gpu.yaml

# Specific GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py
```

## Multi-Node Launch

```bash
# Run this on EACH node (same command, torchrun handles coordination)
torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --rdzv-id=unique_job_id_42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:29500 \
  train.py --config configs/fsdp_large_model.yaml

# MASTER_ADDR = hostname or IP of node 0
# 29500 is conventional; ensure firewall allows it
```

## Reading torchrun Env Vars in Your Script

```python
import os

def get_dist_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return local_rank, rank, world_size, local_world_size

def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1
```

## Elastic Training (Fault Tolerant)

```bash
# min 4, max 8 workers — handles node failures
torchrun \
  --nnodes=4:8 \
  --nproc-per-node=8 \
  --rdzv-id=job_id \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:29500 \
  --max-restarts=3 \
  train.py
```

## NCCL Environment Variables (Set These for Multi-Node)

```bash
export NCCL_DEBUG=INFO           # Verbose NCCL logging (remove in production)
export NCCL_IB_DISABLE=0         # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0   # Specify network interface
export NCCL_TIMEOUT=1800         # Seconds before NCCL gives up (default too low)
export OMP_NUM_THREADS=1         # Prevent CPU oversubscription
```

## SLURM Integration

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=8 \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --config configs/fsdp_large_model.yaml
```
