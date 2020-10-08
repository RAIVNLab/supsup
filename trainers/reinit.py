from functools import partial
from args import args

import torch
import torch.nn as nn

import utils
import numpy as np
import re
import pathlib
import random as r


def cond_cache_masks(
    m,
):
    if hasattr(m, "cache_masks"):
        m.cache_masks()


def init(args):
    pass


def train(
    model, writer, train_loader, optimizer, criterion, epoch, task_idx, data_loader=None
):
    model.zero_grad()
    model.train()

    # Assume we have learned 0 - task_idx - 1 tasks
    model.apply(lambda m: setattr(m, "task", -1))
    model.apply(lambda m: setattr(m, "num_tasks_learned", task_idx))

    train_loader_iter = iter(train_loader)

    # Attempt to adapt
    if task_idx > 0 and epoch == 1:
        model.apply(lambda m: cond_cache_masks(m))
        print("==> Finding closest mask for task", task_idx)

        # Find closest mask from previous n - 1 tasks
        predicted_similar_task = None
        if args.reinit_adapt == "binary":
            predicted_similar_task = binary_entropy_minimization(
                model=model,
                writer=writer,
                criterion=criterion,
                test_loader=train_loader_iter,
                adapt_lr=200,
                num_tasks_learned=task_idx,
            )
        elif args.reinit_adapt == "gradient":
            predicted_similar_task = gradient_entropy_minimization(
                model=model,
                writer=writer,
                criterion=criterion,
                test_loader=train_loader_iter,
                adapt_lr=200,
            )
        elif args.reinit_adapt == "random":
            k = args.reinit_most_recent_k or task_idx
            predicted_similar_task = np.random.randint(max(0, task_idx - k), task_idx)
        elif args.reinit_adapt == "n-1":
            predicted_similar_task = task_idx - 1
        elif args.reinit_adapt == "n-1-imagenet":
            predict_file = None
            files = list(pathlib.Path(args.log_dir).glob("task*"))
            sparse_filter = lambda x: re.search(
                f"task=[0-9]+~sparsity={args.sparsity}", x
            )
            files = [f for f in files if sparse_filter(str(f))]
            files = [f / "final.pt" for f in files if (f / "final.pt").exists()]

            if len(files) > 0:
                predict_file = r.choice(files)
                predicted_similar_task = f"transfer from file {predict_file}"
            else:
                print("=> No prediction file found, starting from scratch")
                predicted_similar_task = "from scratch"

        elif args.reinit_adapt == "running_mean":
            predicted_similar_task = f"0...{task_idx - 1}"

        elif args.reinit_adapt.startswith("starting_from_task_0"):
            predicted_similar_task = 0

        elif args.reinit_adapt.startswith("running_mean_e="):
            predicted_similar_task = f"0...{task_idx - 1}"

        else:
            raise ValueError(f"reinit_adapt value of {args.reinit_adapt} is not valid")

        print(
            f"==> Most similar mask for task {task_idx}"
            f" is {predicted_similar_task} with"
            f" {args.reinit_adapt} adaptation"
        )

        if args.reinit_adapt == "n-1-imagenet":
            if predict_file is not None:
                state_dict = torch.load(predict_file, map_location="cpu")
                for n, p in model.named_parameters():
                    p.data = state_dict[n].clone().to(p.data.device)

                del state_dict
        else:
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    if args.reinit_adapt.startswith("running_mean"):
                        stacked_scores = torch.stack(tuple(m.scores[:task_idx]))
                        stacked_masks = m.stacked[:task_idx]

                        init_scores = stacked_scores.mean(0).data.flatten()
                        init_scores_shape = m.scores[task_idx].shape

                        init_scores_idx = init_scores.flatten().abs().argsort()
                        running_mean_idx = stacked_masks.mean(0).flatten().argsort()

                        m.scores[task_idx].data = init_scores.view(*init_scores_shape)
                        m.scores[task_idx].data *= (
                            stacked_scores.mean(0).norm().detach() / init_scores.norm().detach()
                        )
                    else:
                        m.scores[task_idx].data = m.scores[predicted_similar_task].data

        print(f"=> Training from new initialization")

    model.apply(lambda m: setattr(m, "task", task_idx))

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(target)
        data, target = data.to(args.device), target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            num_samples = batch_idx * len(data)
            num_epochs = len(train_loader.dataset)
            percent_complete = 100.0 * batch_idx / len(train_loader)
            print(
                f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )

            t = (len(train_loader) * epoch + batch_idx) * args.batch_size
            writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)

        if len(train_loader) * (epoch - 1) + batch_idx == args.iter_lim:
            break


def test(model, writer, criterion, test_loader, epoch, task_idx):
    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    logit_entropy = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            if type(data) == list:
                data = data[0]
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            logit_entropy += (
                -(output.softmax(dim=1) * output.log_softmax(dim=1))
                .sum(1)
                .mean()
                .item()
            )
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    logit_entropy /= len(test_loader)
    test_acc = float(correct) / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n")

    writer.add_scalar(f"test/task_{task_idx}/loss", test_loss, epoch)
    writer.add_scalar(f"test/task_{task_idx}/acc", test_acc, epoch)
    writer.add_scalar(f"test/task_{task_idx}/entropy", logit_entropy, epoch)

    return test_acc


def gradient_minimization(
    adaptation_criterion, model, writer, criterion, test_loader, adapt_lr, **kwargs
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    stopping_time = 0

    if args.adapt:
        # First find the alphas!

        # alphas_i contains the "beleif" that the task is i
        alphas = torch.ones(
            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
        )

        for batch_idx, (data_, target) in enumerate(test_loader):
            # Give the model the alphas parameter.
            data, target = data_.to(args.device), target.to(args.device)

            model.apply(lambda m: setattr(m, "alphas", alphas))

            output = model(data)
            adaptation_loss = adaptation_criterion(output, model)

            grad = torch.autograd.grad(adaptation_loss, alphas)
            alphas = alphas - adapt_lr * grad[0]

            max_alpha = (alphas / args.temp).softmax(dim=0).max().item()

            if max_alpha > 0.99:
                stopping_time = batch_idx + 1
                break

        # Now do regular testing with fixed alphas.
        model.apply(lambda m: setattr(m, "alphas", alphas))

    print("Stopping time:", stopping_time)
    model.apply(lambda m: setattr(m, "alphas", None))
    return alphas.argmax()


def binary_minimization(
    adaptation_criterion,
    model,
    writer,
    criterion,
    test_loader,
    adapt_lr,
    num_tasks_learned,
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    predicted_task = -1

    if args.adapt:
        # First find the alphas!

        # alphas_i contains the "belief" that the task is i
        alphas = (
            torch.ones(
                [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
            )
            / num_tasks_learned
        )

        good_inds = torch.arange(args.num_tasks) < num_tasks_learned

        if args.reinit_most_recent_k is not None:
            with torch.no_grad():
                alphas.data[-min(args.reinit_most_recent_k, num_tasks_learned) :] *= (
                    num_tasks_learned / args.reinit_most_recent_k
                )

            good_inds[0 : max(0, num_tasks_learned - args.reinit_most_recent_k)] = False

            print(good_inds)
            print()

        good_inds = good_inds.view(args.num_tasks, 1, 1, 1, 1).to(args.device)

        done = False

        for batch_idx, (data_, target) in enumerate(test_loader):
            if not done:
                data, target = data_.to(args.device), target.to(args.device)

                model.apply(lambda m: setattr(m, "alphas", alphas))

                # Compute the entropy of the logits.
                output = model(data)

                # Take the gradient w.r.t func do one step of SGD on the alphas.
                grad = torch.autograd.grad(adaptation_criterion(output, model), alphas)

                new_alphas = torch.zeros(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device
                )

                inds = grad[0] <= utils.kth_elt(grad[0][good_inds], args.log_base)
                good_inds = inds * good_inds

                new_alphas[good_inds] = 1.0 / good_inds.float().sum().item()
                alphas = new_alphas.clone().detach().requires_grad_(True)
                if good_inds.float().sum() == 1.0:
                    predicted_task = good_inds.flatten().nonzero()[0].item()
                    done = True

        model.apply(lambda m: setattr(m, "alphas", alphas))

    model.apply(lambda m: setattr(m, "alphas", None))

    return predicted_task


def gradient_entropy_minimization(*args, **kwargs):
    def f(logits, model):
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(gradient_minimization, f)(*args, **kwargs)


def binary_entropy_minimization(*args, **kwargs):
    def f(logits, model):
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(binary_minimization, f)(*args, **kwargs)
