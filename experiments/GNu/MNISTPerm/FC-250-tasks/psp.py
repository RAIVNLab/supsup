import sys, os

sys.path.append(os.path.abspath("."))
from main import main as run
from args import args


def main():
    args.set = 'MNISTPerm'
    args.multigpu = [0]
    args.model = 'FC1024'
    args.conv_type = 'PSPRotation'
    args.conv_init = 'xavier_normal'
    args.name = f'id=psprot_fc_mnistperm'

    args.adapt_lrs = [0]
    args.eval_ckpts = [10, 50, 100, 150, 200]
    args.num_tasks = 250
    args.adaptor = "gt"
    args.hard_alphas = True
    args.output_size = 10

    args.train_weight_tasks = -1
    args.train_weight_lr = 1e-4
    args.lr = 1e-4
    args.momentum = 0.5
    args.optimizer = 'rmsprop'

    args.no_scheduler = True
    args.iter_lim = 1000
    args.epochs = 3
    args.ortho_group = True
    args.save = True

    args.data = '~/data'
    args.log_dir = "~/checkpoints/test"

    run()


if __name__ == "__main__":
    main()
