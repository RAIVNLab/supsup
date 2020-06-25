from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))


def kwargs_to_cmd(kwargs):
    cmd = "python main.py "
    for flag, val in kwargs.items():
        cmd += f"--{flag}={val} "

    return cmd


def run_exp(gpu_num, in_queue):
    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return

        before = time.time()

        experiment["multigpu"] = gpu_num
        print(f"==> Starting experiment {kwargs_to_cmd(experiment)}")
        os.system(kwargs_to_cmd(experiment))


        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=1, type=int)
    parser.add_argument('--data', default='~/data', type=str)
    args = parser.parse_args()

    gpus = args.gpu_sets
    data = args.data

    config = "experiments/GG/splitimagenet/configs/rn50-supsup-adam.yaml"
    log_dir = "runs/splitimagenet-rn50-supsup"
    sparsities = [8, 4, 16, 32] # We report [4, 8, 16] in the paper
    experiments = []

    
    for sparsity, task_idx in product(sparsities, range(100)):
        kwargs = {
            "config": config,
            "name": f"id=rn50-supsup~task={task_idx}~sparsity={sparsity}",
            "sparsity": sparsity,
            "task-eval": task_idx,
            "log-dir": log_dir,
            "data": data,
        }

        experiments.append(kwargs)

    print(experiments)
    input("Press any key to continue...")
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
