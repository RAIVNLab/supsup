from args import args

import torch
import torch.nn as nn


def init(args):
    pass


def train(model, writer, train_loader, optimizer, criterion, epoch, task_idx, data_loader=None):
    model.zero_grad()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.iter_lim < 0 or len(train_loader) * (epoch - 1) + batch_idx < args.iter_lim:
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
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
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
