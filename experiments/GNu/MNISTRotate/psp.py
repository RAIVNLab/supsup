import sys, os
sys.path.append(os.path.abspath('.'))
from main import main as run
from args import args

def main():
    args.set = 'RotatingMNIST'
    args.num_tasks = 36

    args.multigpu = [0]
    args.model = 'FC1024'
    args.conv_type = 'PSPRotation'
    args.conv_init = 'xavier_normal'

    args.adaptor = "gt"
    args.epochs = 3

    args.no_scheduler = True
    args.iter_lim = 1000
    args.eval_ckpts = [2, 5, 10, 15, 20, 25, 30, 35]

    args.name = f'id=psp_mnistrotate'
    args.output_size = 10

    args.train_weight_tasks = -1
    args.train_weight_lr = 1e-4
    args.lr = 1e-4
    args.momentum = 0.5
    args.optimizer = 'rmsprop'
    args.ortho_group = True

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()

if __name__ == '__main__':
    main()