import sys, os

sys.path.append(os.path.abspath("."))
from main import main as run
from args import args


def main():
    args.data_to_repeat = 1
    args.set = 'MNISTPerm'
    args.multigpu = [0]
    args.model = 'FC1024'
    args.conv_type = 'FastMultitaskMaskConv'
    args.conv_init = 'signed_constant'

    args.eval_ckpts = [10, 50, 100, 150, 200]
    args.num_tasks = 250
    args.adaptor = "se_oneshot_entropy_minimization"
    args.output_size = 500

    args.name = f'id=supsup_h_fc_mnistperm'

    args.lr = 1e-4
    args.optimizer = 'rmsprop'
    args.iter_lim = 1000
    args.epochs = 3

    args.save = True
    args.no_scheduler = True
    args.unshared_labels = True
    args.data_to_repeat = 1

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()


if __name__ == "__main__":
    main()
