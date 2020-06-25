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
    seeds = list(range(args.seeds))
    data = args.data

    config = "experiments/GG/splitcifar100/configs/rn18-supsup-transfer.yaml"
    log_dir = "runs/rn18-supsup-transfer"
    experiments = []
    sparsities = [1, 2, 4, 8, 16, 32] # Higher sparsities = More dense subnetworks

    for sparsity, seed in product(sparsities, seeds):
        kwargs = {
            "config": config,
            "name": f"id=supsup-transfer~seed={seed}~sparsity={sparsity}",
            "sparsity": sparsity,
            "seed": seed,
            "log-dir": log_dir,
            "epochs": 250,
            "reinit-adapt": "running_mean",
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
