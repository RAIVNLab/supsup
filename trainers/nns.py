from args import args

import torch
import torch.nn as nn

import numpy as np

k = 100
def init(args):
    pass



def train(model, writer, train_loader, optimizer, criterion, epoch, task_idx, data_loader=None):
    model.zero_grad()
    model.train()

    maxn = 5

    for batch_idx, (data, target) in enumerate(train_loader):

        if model.task >= args.num_tasks:
            continue

        data, target = data.to(args.device), target.to(args.device)

        if (batch_idx % k) == 0:

            if not hasattr(model, 'task_total'):
                model.task_total = 1
                ind = 0
                value = 0
                grad_entropy = 0
            else:

                model.zero_grad()
                model.apply(lambda m: setattr(m, "task", -1))

                alphas = (
                        torch.ones(
                            [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True
                        )
                        / max(model.task_total, maxn)
                )

                model.apply(lambda m: setattr(m, "num_tasks_learned", max(model.task_total, maxn)))
                model.apply(lambda m: setattr(m, "alphas", alphas))
                output = model(data)
                logit_entropy = -(output.softmax(dim=1) * output.log_softmax(dim=1)).sum(1).mean()
                grad = torch.autograd.grad(logit_entropy, alphas)[0]
                grad = -grad.flatten()[:model.num_tasks_learned]

                #p = (grad * 2 * model.num_tasks_learned).softmax(dim=0)
                #p = (10 * np.log(model.num_tasks_learned) * grad).softmax(dim=0)
                p = grad.softmax(dim=0).max()

                #grad_entropy = #p - 1./model.num_tasks_learned #-( p * p.log() ).sum()
                grad_entropy = p
                _, ind = grad.max(dim=0)
                #value = 1.2/model.num_tasks_learned
                # seems like optimal value here is somewhere between 1.1 and 1.2
                value = 1.125/model.num_tasks_learned

                ind = ind.item()

                if grad_entropy < value:
                    model.task_total += 1
                    ind = model.task_total - 1
                    print('NEW TASK', ind)

            model.apply(lambda m: setattr(m, "task", ind))
            if model.task >= args.num_tasks:
                continue

        optimizer.zero_grad()
        model.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % k == 0:
            num_samples = batch_idx * len(data)
            num_epochs = len(train_loader.dataset)
            percent_complete = 100.0 * batch_idx / len(train_loader)
            print(
                f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Ind: {ind}\t"
                f"GE: {grad_entropy:.4f}\t"
                f"V: {value:.4f}"
            )
            #print(grad.flatten())

            t = (len(train_loader) * epoch + batch_idx) * args.batch_size
            writer.add_scalar(f"train_{task_idx}/loss", loss.item(), t)

        if len(train_loader) * (epoch - 1) + batch_idx == args.iter_lim:
            break


def test(model, writer, criterion, test_loader, epoch, task_idx):
    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    logit_entropy = 0.0
    test_acc = 0

    if model.task < args.num_tasks:

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

        test_loss /= len(test_loader.dataset)
        logit_entropy /= len(test_loader.dataset)
        test_acc = float(correct) / len(test_loader.dataset)

        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f}%)\n")

        writer.add_scalar(f"test_{task_idx}/loss", test_loss, epoch)
        writer.add_scalar(f"test_{task_idx}/acc", test_acc, epoch)
        writer.add_scalar(f"test_{task_idx}/entropy", logit_entropy, epoch)

    return test_acc
