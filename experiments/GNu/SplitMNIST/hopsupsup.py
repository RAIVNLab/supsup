import sys, os
sys.path.append(os.path.abspath('.'))
from main import main as run
from args import args
import numpy as np

def main():
    args.set = 'PartitionMNISTV2'
    args.num_tasks = 5
    args.multigpu = [0]
    args.model = 'BNNet'
    args.conv_type = 'StandardConv'
    args.bn_type = 'FastHopMaskBN'
    args.conv_init = 'signed_constant'
    args.adaptor = 'hopfield_recovery'
    args.gamma = 1.5e-3
    args.batch_size = 128
    args.test_batch_size = 64
    args.epochs = 1
    args.adapt = True
    args.output_size = 10
    args.eval_ckpts = []

    args.name = 'id=hopsupsup'

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()

if __name__ == '__main__':
    main()