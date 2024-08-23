import os
import time  # 引入时间模块
import argparse
import random
import numpy as np
import torch
import torch_mlu
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank",type=int,default=0, help="local_rank")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    #if 'LOCAL_RANK' not in os.environ:
    #    os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    """
    设置随机种子以确保结果的可重复性。

    为了在分布式训练中确保每个节点上的随机性一致，我们需要根据配置文件中的种子值和当前节点的rank来设置随机种子。
    通过这种方式，尽管每个节点可能生成不同的随机数，但这种差异是由配置决定的，从而保证了实验的可比较性和一定程度的可重复性。

    参数:
    - config: 配置对象，包含运行配置信息如种子值等。

    返回值:
    无。该函数的目的是为了设置随机种子，不返回任何值。
    """
    # 根据配置文件中的种子值和当前节点的rank计算随机种子
    seed = config.run_cfg.seed + get_rank()

    # 设置Python、NumPy和PyTorch的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 禁止cuDNN使用heuristics来选择最佳的卷积方法，以确保实验的可重复性
    cudnn.benchmark = False
    # 设置cuDNN具有确定性，以确保卷积计算是可重复的
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    从配置文件中获取runner类。默认情况下使用基于epoch的runner。
    
    参数:
    - cfg: 配置对象，应包含运行器的配置信息。
    
    返回:
    - runner_cls: 从配置中获取的runner类。
    """
    # 根据配置文件中的runner类型获取对应的runner类
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    # 返回获取到的runner类
    return runner_cls


def main():
    # 在使用 NCCL 后端时，允许在主进程中完成自动下载，而不会超时。
    # 通过设置环境变量 "NCCL_BLOCKING_WAIT" = "1" 来实现。

    # 在 init_distributed_mode() 之前设置，以确保所有 rank 共享相同的 job_id。
    job_id = now()

    # 解析命令行参数，并返回配置对象。
    cfg = Config(parse_args())

    # 初始化分布式模式，根据运行配置。
    init_distributed_mode(cfg.run_cfg)

    # 设置随机种子，以确保分布式训练的一致性。
    setup_seeds(cfg)

    # 在 init_distributed_mode() 之后设置，仅在主进程上进行日志记录。
    setup_logger()

    # 打印配置信息，便于调试和记录。
    cfg.pretty_print()

    # 根据配置设置任务。
    task = tasks.setup_task(cfg)
    # 构建数据集。
    datasets = task.build_datasets(cfg)
    # 构建模型。
    model = task.build_model(cfg)

    # 根据配置获取训练运行器类，并初始化。
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    # # 开始训练。
    # runner.train()
    # 记录训练开始时间
    start_time = time.time()

    # 开始训练
    runner.train()

    # 记录训练结束时间
    end_time = time.time()

    # 计算并输出总耗时
    total_time = end_time - start_time
    print(f"模型训练完成，总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")


if __name__ == "__main__":
    main()
