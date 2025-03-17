'''
假设有以下场景, 一共8个进程, [0,1,2,3]和[4,5,6,7]组成一个group,
其中[0,4],[1,5],[2,6],[3,7]又分别组成一个组，
请使用 torch.distributed 完成上述进程组的创建和管理
'''

import os
import torch
import torch.distributed as dist
from multiprocessing import Process

def run(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:12345',  # 使用 TCP 初始化方法
        rank=rank,
        world_size=world_size
    )

    print(f"Rank {rank} is initialized.")

    # 创建子进程组
    # 全局进程组
    global_group = dist.new_group(ranks=list(range(world_size)))
    # print(global_group)
    # 子进程组 [0, 1, 2, 3]
    subgroup1 = dist.new_group(ranks=[0, 1, 2, 3])

    # 子进程组 [4, 5, 6, 7]
    subgroup2 = dist.new_group(ranks=[4, 5, 6, 7])

    # 子进程组 [0, 4], [1, 5], [2, 6], [3, 7]
    subgroup3 = []
    for i in range(4):
        subgroup3.append(dist.new_group(ranks=[i, i + 4]))

    # 在 subgroup1 中进行广播（仅 rank 0-3 参与）
    if rank in [0, 1, 2, 3]:
        tensor = torch.tensor([rank], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=subgroup1)
        print(f"Rank {rank} received tensor {tensor} from subgroup1")

    # 在 subgroup3[0] 中进行广播（仅 rank 0 和 4 参与）
    if rank in [0, 4]:
        tensor = torch.tensor([rank], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=subgroup3[0])
        print(f"Rank {rank} received tensor {tensor} from subgroup3[0]")

    # 销毁进程组
    dist.destroy_process_group()

def main():
    world_size = 8
    processes = []

    # 创建 8 个进程
    for rank in range(world_size):
        p = Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == "__main__":
    # 设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    main()