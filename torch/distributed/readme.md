# torch.distributed
本质上 torch.distributed 是为了建立进程组之间的通信，这些进程可能处于不同的机器。

1. 核心概念

进程组（Process Group）: 用于管理一组进程之间的通信。每个进程组有一个唯一的 ID，进程可以通过这个组进行通信。

Rank: 进程在进程组中的唯一标识符（从 0 开始）。

World Size: 进程组中的总进程数。

Local Rank: 当前节点上的进程标识符（从 0 开始）。

Backend: 通信后端，如 nccl（适用于 GPU）、gloo（适用于 CPU）等。

初始化函数：
```python
import torch.distributed as dist

def init_distributed(backend='nccl', world_size=1, rank=0, master_addr='localhost', master_port='12355'):
    dist.init_process_group(
        backend=backend,       # 通信后端
        init_method=f'tcp://{master_addr}:{master_port}',  # 初始化方法
        world_size=world_size,  # 总进程数
        rank=rank               # 当前进程的 rank
    )
```
init_method 的作用是`告诉整个进程组如何找到主节点并建立通信`。它定义了分布式训练中各个进程如何初始化和连接到主节点，从而协调整个进程组的通信.