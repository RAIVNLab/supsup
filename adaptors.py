from args import args
import torch
from torch import optim
import math

import numpy as np
import pathlib

from models.modules import FastHopMaskBN
from models import module_util
from utils import kth_elt
from functools import partial


def adapt_test(
    model,
    test_loader,
    alphas=None,
):
    correct = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if alphas is not None:
                model.apply(lambda m: setattr(m, "alphas", alphas))
            
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = float(correct) / len(test_loader.dataset)

        print(
            f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
        )

    return test_acc

# gt means ground truth task -- corresponds to GG
def gt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()

    alphas = (
        torch.zeros(
            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
        )
    )
    alphas[task] = 1


    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    test_acc = adapt_test(
        model,
        test_loader,
        alphas,
    )

    model.apply(lambda m: setattr(m, "alphas", None))
    return test_acc


# The oneshot minimization algorithm.
def se_oneshot_minimization(
    adaptation_criterion,
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    correct = 0
    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        denominator = model.num_tasks_learned if args.trainer and "nns" in args.trainer else num_tasks_learned
        # alphas_i contains the "beleif" that the task is i
        alphas = (
                torch.ones(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
                )
                / denominator
        )

        model.apply(lambda m: setattr(m, "alphas", alphas))

        # Compute the output
        output = model(data)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        # Take the gradient w.r.t objective
        grad = torch.autograd.grad(adaptation_criterion(output, model), alphas)
        value, ind = grad[0].min(dim=0)
        alphas = torch.zeros([args.num_tasks, 1, 1, 1, 1], device=args.device)
        alphas[ind] = 1

        print(ind)
        predicted_task = ind.item()
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue

        # Now do regular testing with inferred task.
        model.apply(lambda m: setattr(m, "alphas", alphas))

        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = float(correct) / len(test_loader.dataset)

    print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    model.apply(lambda m: setattr(m, "alphas", None))

    return test_acc


# The binary minimization algorithm.
def se_binary_minimization(
    adaptation_criterion,
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    correct = 0
    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)

        # alphas_i contains the "beleif" that the task is i
        alphas = (
                torch.ones(
                    [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
                )
                / num_tasks_learned
        )
        # store the "good" indecies, i.e. still valid for optimization
        good_inds = torch.arange(args.num_tasks) < num_tasks_learned
        good_inds = good_inds.view(args.num_tasks, 1, 1, 1, 1).to(args.device)
        done = False

        prevent_inf_loop_iter = 0
        while not done:
            prevent_inf_loop_iter += 1
            if prevent_inf_loop_iter > np.log2(args.num_tasks) + 1:
                print('InfLoop')
                break
            model.zero_grad()

            model.apply(lambda m: setattr(m, "alphas", alphas))

            # Compute the output.
            output = model(data)

            # Take the gradient w.r.t objective
            grad = torch.autograd.grad(adaptation_criterion(output, model), alphas)

            new_alphas = torch.zeros([args.num_tasks, 1, 1, 1, 1], device=args.device)

            inds = grad[0] <= kth_elt(grad[0][good_inds], args.log_base)
            good_inds = inds * good_inds
            new_alphas[good_inds] = 1.0 / good_inds.float().sum().item()
            alphas = new_alphas.clone().detach().requires_grad_(True)
            if good_inds.float().sum() == 1.0:
                predicted_task = good_inds.flatten().nonzero()[0].item()
                done = True

        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue


        model.apply(lambda m: setattr(m, "alphas", alphas))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_acc = float(correct) / len(test_loader.dataset)
    task_correct = float(task_correct) / len(test_loader.dataset)

    print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    model.apply(lambda m: setattr(m, "alphas", None))

    return test_acc

# ABatchE using entropy objective.
def se_be_adapt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    test_loss = 0
    correct = 0
    data_to_repeat = args.data_to_repeat

    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        model.apply(lambda m: setattr(m, "task", -1))

        if data.shape[0] >= data_to_repeat:
            rep_data = torch.cat(
                tuple([
                    data[j].unsqueeze(0).repeat(model.num_tasks_learned, 1, 1, 1)
                    for j in range(data_to_repeat)
                ]),
                dim=0
            )

            logits = model(rep_data)

            ent = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1)
            ent_reshape = ent.view(data_to_repeat, num_tasks_learned)
            ent_reshape_mean = ent_reshape.mean(dim=0)
            v, i = ent_reshape_mean.min(dim=0)

            ind = i.item()
            print(ind)
        predicted_task = ind
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue
        model.apply(lambda m: setattr(m, "task", ind))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_acc = float(correct) / len(test_loader.dataset)
    task_correct = float(task_correct) / len(test_loader.dataset)

    print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    return test_acc

# ABatchE using M objective.
def se_be_max_adapt(
    model,
    writer,
    test_loader,
    num_tasks_learned,
    task,
):
    model.zero_grad()
    model.train()
    correct = 0

    data_to_repeat = args.data_to_repeat

    task_correct = 0

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        model.apply(lambda m: setattr(m, "task", -1))
        rep_data = torch.cat(
            tuple([
                data[j].unsqueeze(0).repeat(model.num_tasks_learned, 1, 1, 1)
                for j in range(data_to_repeat)
            ]),
            dim=0
        )

        logits = model(rep_data)
        sm = logits.softmax(dim=1)
        ent, _ = sm.max(dim=1)
        ent_reshape = ent.view(data_to_repeat, num_tasks_learned)
        ent_reshape_mean = ent_reshape.mean(dim=0)

        v, i = ent_reshape_mean.max(dim=0)
        ind = i.item()
        print(ind)
        predicted_task = ind
        if predicted_task == task:
            task_correct += 1
        else:
            if args.unshared_labels:
                continue
        model.apply(lambda m: setattr(m, "task", ind))


        with torch.no_grad():

            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = float(correct) / len(test_loader.dataset)

    print(
        f"\nTest set: Accuracy: ({test_acc:.4f}%)\n"
    )

    return test_acc


def se_oneshot_entropy_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(se_oneshot_minimization, f)(*arg, **kwargs)

def se_oneshot_g_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        m = (torch.arange(logits.size(1)) < args.real_neurons).float().unsqueeze(0).to(args.device)
        logits = (logits * m).detach() + logits * (1-m)
        return logits.logsumexp(dim=1).mean()

    return partial(se_oneshot_minimization, f)(*arg, **kwargs)

def se_binary_entropy_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    return partial(se_binary_minimization, f)(*arg, **kwargs)

def se_binary_g_minimization(*arg, **kwargs):
    def f(logits, model):
        logits = logits[:args.data_to_repeat]
        m = (torch.arange(logits.size(1)) < args.real_neurons).float().unsqueeze(0).to(args.device)
        logits = (logits * m).detach() + logits * (1 - m)
        return logits.logsumexp(dim=1).mean()

    return partial(se_binary_minimization, f)(*arg, **kwargs)

# HopSupSup -- Hopfield recovery.
def hopfield_recovery(
    model, writer, test_loader, num_tasks_learned, task,
):
    model.zero_grad()
    model.train()
    # stopping time tracks how many epochs were required to adapt.
    stopping_time = 0
    correct = 0
    taskname = f"{args.set}_{task}"


    params = []
    for n, m in model.named_modules():
        if isinstance(m, FastHopMaskBN):
            out = torch.stack(
                [
                    2 * module_util.get_subnet_fast(m.scores[j]) - 1
                    for j in range(m.num_tasks_learned)
                ]
            )

            m.score = torch.nn.Parameter(out.mean(dim=0))
            params.append(m.score)

    optimizer = optim.SGD(
        params, lr=500, momentum=args.momentum, weight_decay=args.wd,
    )

    for batch_idx, (data_, target) in enumerate(test_loader):
        data, target = data_.to(args.device), target.to(args.device)
        hop_loss = None

        for n, m in model.named_modules():
            if isinstance(m, FastHopMaskBN):
                s = 2 * module_util.GetSubnetFast.apply(m.score) - 1
                target = 2 * module_util.get_subnet_fast(m.scores[task]) - 1
                distance = (s != target).sum().item()
                writer.add_scalar(
                    f"adapt_{taskname}/distance_{n}",
                    distance,
                    batch_idx + 1,
                )
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data)
        logit_entropy = (
            -(output.softmax(dim=1) * output.log_softmax(dim=1)).sum(1).mean()
        )
        for n, m in model.named_modules():
            if isinstance(m, FastHopMaskBN):
                s = 2 * module_util.GetSubnetFast.apply(m.score) - 1
                if hop_loss is None:
                    hop_loss = (
                        -0.5 * s.unsqueeze(0).mm(m.W.mm(s.unsqueeze(1))).squeeze()
                    )
                else:
                    hop_loss += (
                        -0.5 * s.unsqueeze(0).mm(m.W.mm(s.unsqueeze(1))).squeeze()
                    )

        hop_lr = args.gamma * (
            float(batch_idx + 1) / len(test_loader)
        )
        hop_loss =  hop_lr * hop_loss
        ent_lr = 1 - (float(batch_idx + 1) / len(test_loader))
        logit_entropy = logit_entropy * ent_lr
        (logit_entropy + hop_loss).backward()
        optimizer.step()

        writer.add_scalar(
            f"adapt_{taskname}/{num_tasks_learned}/entropy",
            logit_entropy.item(),
            batch_idx + 1,
        )

        writer.add_scalar(
            f"adapt_{taskname}/{num_tasks_learned}/hop_loss",
            hop_loss.item(),
            batch_idx + 1,
        )

    test_acc = adapt_test(
        model,
        test_loader,
        alphas=None,
    )

    model.apply(lambda m: setattr(m, "alphas", None))
    return test_acc
