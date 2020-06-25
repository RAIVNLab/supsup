import os
import pathlib
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from args import args
import adaptors
import data
import schedulers
import trainers
import utils

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict



def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "settings.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")

    # Get dataloader.
    data_loader = getattr(data, args.set)()

    # Track accuracy on all tasks.
    if args.num_tasks:
        best_acc1 = [0.0 for _ in range(args.num_tasks)]
        curr_acc1 = [0.0 for _ in range(args.num_tasks)]
        adapt_acc1 = [0.0 for _ in range(args.num_tasks)]

    # Get the model.
    model = utils.get_model()

    # If necessary, set the sparsity of the model of the model using the ER sparsity budget (see paper).
    if args.er_sparsity:
        for n, m in model.named_modules():
            if hasattr(m, "sparsity"):
                m.sparsity = min(
                    0.5,
                    args.sparsity
                    * (m.weight.size(0) + m.weight.size(1))
                    / (
                        m.weight.size(0)
                        * m.weight.size(1)
                        * m.weight.size(2)
                        * m.weight.size(3)
                    ),
                )
                print(f"Set sparsity of {n} to {m.sparsity}")

    # Put the model on the GPU,
    model = utils.set_gpu(model)

    # Optionally resume from a checkpoint.
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(
                args.resume, map_location=f"cuda:{args.multigpu[0]}"
            )
            best_acc1 = checkpoint["best_acc1"]
            pretrained_dict = checkpoint["state_dict"]
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict)

            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=run_base_dir)

    # Track the number of tasks learned.
    num_tasks_learned = 0

    trainer = getattr(trainers, args.trainer or "default")
    print(f"=> Using trainer {trainer}")

    train, test = trainer.train, trainer.test

    # Initialize model specific context (editorial note: avoids polluting main file)
    if hasattr(trainer, "init"):
        trainer.init(args)

    # TODO: Put this in another file
    if args.task_eval is not None:
        assert 0 <= args.task_eval < args.num_tasks, "Not a valid task idx"
        print(f"Task {args.set}: {args.task_eval}")

        model.apply(lambda m: setattr(m, "task", args.task_eval))

        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(args.task_eval)

        optimizer = get_optimizer(args, model)
        lr_scheduler = schedulers.get_policy(args.lr_policy or "cosine_lr")(
            optimizer, args
        )

        # Train and do inference and normal for args.epochs epcohs.
        best_acc1 = 0.0

        for epoch in range(0, args.epochs):
            lr_scheduler(epoch, None)

            train(
                model,
                writer,
                data_loader.train_loader,
                optimizer,
                criterion,
                epoch,
                task_idx=args.task_eval,
                data_loader=None,
            )

            curr_acc1 = test(
                model,
                writer,
                criterion,
                data_loader.val_loader,
                epoch,
                task_idx=args.task_eval,
            )

            if curr_acc1 > best_acc1:
                best_acc1 = curr_acc1

        utils.write_result_to_csv(
            name=f"{args.name}~{args.set}~task={args.task_eval}",
            curr_acc1=curr_acc1,
            best_acc1=best_acc1,
            save_dir=run_base_dir,
        )

        if args.save:
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / "final.pt",
            )

        return best_acc1

    # Iterate through all tasks.
    for idx in range(args.num_tasks or 0):
        print(f"Task {args.set}: {idx}")

        # Tell the model which task it is trying to solve -- in Scenario NNs this is ignored.
        model.apply(lambda m: setattr(m, "task", idx))

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx)

        # Clear the grad on all the parameters.
        for p in model.parameters():
            p.grad = None

        # Make a list of the parameters relavent to this task.
        params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            split = n.split(".")
            if split[-2] in ["scores", "s", "t"] and (
                int(split[-1]) == idx or (args.trainer and "nns" in args.trainer)
            ):
                params.append(p)
            # train all weights if train_weight_tasks is -1, or num_tasks_learned < train_weight_tasks
            if (
                args.train_weight_tasks < 0
                or num_tasks_learned < args.train_weight_tasks
            ):
                if split[-1] == "weight" or split[-1] == "bias":
                    params.append(p)

        # train_weight_tasks specifies the number of tasks that the weights are trained for.
        # e.g. in SupSup, train_weight_tasks = 0. in BatchE, train_weight_tasks = 1.
        # If training weights, use train_weight_lr. Else use lr.
        lr = (
            args.train_weight_lr
            if args.train_weight_tasks < 0
            or num_tasks_learned < args.train_weight_tasks
            else args.lr
        )

        # get optimizer, scheduler
        if args.optimizer == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=args.wd)
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr)
        else:
            optimizer = optim.SGD(
                params, lr=lr, momentum=args.momentum, weight_decay=args.wd
            )
        if args.no_scheduler:
            scheduler = None
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Train on the current task.
        for epoch in range(1, args.epochs + 1):
            train(
                model,
                writer,
                data_loader.train_loader,
                optimizer,
                criterion,
                epoch,
                idx,
                data_loader,
            )

            # Required for our PSP implementation, not used otherwise.
            utils.cache_weights(model, num_tasks_learned + 1)

            curr_acc1[idx] = test(
                model, writer, criterion, data_loader.val_loader, epoch, idx
            )
            if curr_acc1[idx] > best_acc1[idx]:
                best_acc1[idx] = curr_acc1[idx]
            if scheduler:
                scheduler.step()

            if (
                args.iter_lim > 0
                and len(data_loader.train_loader) * epoch > args.iter_lim
            ):
                break

        # Save memory by deleting the optimizer and scheduler.
        del optimizer, scheduler, params

        # Increment the number of tasks learned.
        num_tasks_learned += 1

        # If operating in NNS scenario, get the number of tasks learned count from the model.
        if args.trainer and "nns" in args.trainer:
            model.apply(
                lambda m: setattr(
                    m, "num_tasks_learned", min(model.num_tasks_learned, args.num_tasks)
                )
            )
        else:
            model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))

        # TODO series of asserts with required arguments (eg num_tasks)
        # args.eval_ckpts contains values of num_tasks_learned for which testing on all tasks so far is performed.
        # this is done by default when all tasks have been learned, but you can do something like
        # args.eval_ckpts = [5,10] to also do this when 5 tasks are learned, and again when 10 tasks are learned.
        if num_tasks_learned in args.eval_ckpts or num_tasks_learned == args.num_tasks:
            avg_acc = 0.0
            avg_correct = 0.0

            # Settting task to -1 tells the model to infer task identity instead of being given the task.
            model.apply(lambda m: setattr(m, "task", -1))

            # an "adaptor" is used to infer task identity.
            # args.adaptor == gt implies we are in scenario GG.

            # This will cache all of the information the model needs for inferring task identity.
            if args.adaptor != "gt":
                utils.cache_masks(model)

            # Iterate through all tasks.
            adapt = getattr(adaptors, args.adaptor)

            for i in range(num_tasks_learned):
                print(f"Testing {i}: {args.set} ({i})")
                # model.apply(lambda m: setattr(m, "task", i))

                # Update the data loader so it is returning data for the right task.
                data_loader.update_task(i)

                # Clear the stored information -- memory leak happens if not.
                for p in model.parameters():
                    p.grad = None

                for b in model.buffers():
                    b.grad = None

                torch.cuda.empty_cache()

                adapt_acc = adapt(
                    model,
                    writer,
                    data_loader.val_loader,
                    num_tasks_learned,
                    i,
                )

                adapt_acc1[i] = adapt_acc
                avg_acc += adapt_acc

                torch.cuda.empty_cache()
                utils.write_adapt_results(
                    name=args.name,
                    task=f"{args.set}_{i}",
                    num_tasks_learned=num_tasks_learned,
                    curr_acc1=curr_acc1[i],
                    adapt_acc1=adapt_acc,
                    task_number=i,
                )

            writer.add_scalar(
                "adapt/avg_acc", avg_acc / num_tasks_learned, num_tasks_learned
            )
            
            utils.clear_masks(model)
            torch.cuda.empty_cache()


    if args.save:
        torch.save(
            {
                "epoch": args.epochs,
                "arch": args.model,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "curr_acc1": curr_acc1,
                "args": args,
            },
            run_base_dir / "final.pt",
        )

    for idx in range(args.num_tasks):
        utils.write_result_to_csv(
            name=args.name + f"~task={args.set}_{idx}",
            curr_acc1=curr_acc1[idx],
            best_acc1=best_acc1[idx],
            save_dir=run_base_dir,
        )

    return adapt_acc1


# TODO: Remove this with task-eval
def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": args.wd,},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    return optimizer


if __name__ == "__main__":
    main()
