# Multi-Node Distributed Training

## Launch Pattern

```bash
# Node 0 (master):
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="10.0.0.1:29500" \
  train.py

# Node 1, 2, 3 (change --node_rank):
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=1 \   # increment per node
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  ...
```

## Required NCCL Environment Variables

```bash
# Set on ALL nodes before launching:
export NCCL_DEBUG=INFO              # enable during debug, disable in prod
export NCCL_IB_DISABLE=0           # 0 = use InfiniBand (set 1 if no IB)
export NCCL_NET_GDR_LEVEL=2        # GPU Direct RDMA (if supported)
export NCCL_SOCKET_IFNAME=eth0     # network interface (check with `ip link`)
export NCCL_TIMEOUT=1800           # 30 min timeout (increase for large models)

# For AWS/GCP with EFA/RDMA:
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
```

## Connectivity Test (run before training)

```bash
# Test NCCL connectivity between nodes:
python -c "
import torch.distributed as dist
import torch
dist.init_process_group('nccl')
rank = dist.get_rank()
t = torch.ones(1).cuda()
dist.all_reduce(t)
print(f'Rank {rank}: all_reduce OK, result={t.item()}')
dist.destroy_process_group()
"
```

If this hangs → NCCL connectivity issue. Check firewall, NCCL_SOCKET_IFNAME, IB config.

## Health Check at Startup

```python
def verify_distributed_setup():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # All-reduce a test tensor to verify connectivity
    test = torch.ones(1, device="cuda")
    dist.all_reduce(test)
    assert test.item() == world_size, f"Rank {rank}: all_reduce failed"

    if rank == 0:
        print(f"Distributed setup OK: world_size={world_size}")
```

## Gradient Compression (optional, for bandwidth-limited setups)

```python
# PowerSGD — reduces gradient communication, slight accuracy cost
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
state = powerSGD.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,
    warm_start=True,
)
model.register_comm_hook(state, powerSGD.powerSGD_hook)
```

Use only when: inter-node bandwidth < 25 Gbps AND model gradients are large.
