import sys, os
sys.path.append(os.path.abspath('.'))
from main import main as run
from args import args

def main():
    args.set = 'RotatingMNIST'
    args.num_tasks = 36

    args.multigpu = [0]
    args.model = 'FC1024'
    args.conv_type = 'FastMultitaskMaskConv'
    args.conv_init = 'signed_constant'

    args.eval_ckpts = [2, 5, 10, 15, 20, 25, 30, 35]
    args.name = f'id=supsup_mnistrotate'
    args.adaptor = "se_binary_entropy_minimization"

    args.lr = 1e-4
    args.optimizer = 'rmsprop'

    args.iter_lim = 1000
    args.epochs = 3
    args.data_to_repeat = 128
    args.output_size = 200
    args.save = True
    args.no_scheduler = True
    args.unshared_labels = True

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()

if __name__ == '__main__':
    main()