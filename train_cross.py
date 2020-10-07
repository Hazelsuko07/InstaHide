#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os

import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import random

import models
from utils import progress_bar, chunks, save_fig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(
    description='PyTorch InstaHide Training, CIFAR-10')

# Training configurations
parser.add_argument('--model',
                    default="ResNet18",
                    type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--data', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--nclass', default=10, type=int,
                    help='number of classes')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch',
                    default=200,
                    type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment',
                    dest='augment',
                    action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')

# Saving configurations
parser.add_argument('--name', default='cross', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--help_dir',
                    default='./data/imagenet_filter_40', type=str)

# InstaHide configurations
parser.add_argument('--klam', default=4, type=int, help='number of lambdas')
parser.add_argument('--mode', default='instahide',
                    type=str, help='InsatHide or Mixup')
parser.add_argument('--pair', action='store_true')
parser.add_argument('--upper', default=0.65, type=float, help='the upper bound of any coefficient')
parser.add_argument('--dom', default=0.3, type=float, help='the lower bound of the sum of coefficients of two private images')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
best_acc = 0  # best test accuracy

## --------------- Functions for train & eval --------------- ##


def label_to_onehot(target, num_classes=args.nclass):
    '''Returns one-hot embeddings of scaler labels'''
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def mixup_criterion(pred, ys, lam_batch, num_class=args.nclass):
    '''Returns mixup loss'''
    if args.pair:
        inside_cnt = 2
    else:
        inside_cnt = (args.klam+1)//2
    ys_onehot = [label_to_onehot(y, num_classes=num_class) for y in ys]
    mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
    # for i in range(1, args.klam):
    for i in range(1, inside_cnt):
        mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])
    l = cross_entropy_for_onehot(pred, mixy)
    return l


def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res


def mixup_data(x, y, x_help, use_cuda=True):
    '''Returns mixed inputs, lists of targets, and lambdas'''
    lams = np.random.normal(0, 1, size=(x.size()[0], args.klam))
    for i in range(x.size()[0]):
        lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
        if args.klam > 1:
            while lams[i].max() > args.upper:     # upper bounds a single lambda
                lams[i] = np.random.normal(0, 1, size=(1, args.klam))
                lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
            if args.dom > 0:
                while (lams[i][0] + lams[i][1]) < args.dom:
                    lams[i] = np.random.normal(0, 1, size=(1, args.klam))
                    lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))

    lams = torch.from_numpy(lams).float().to(device)

    mixed_x = vec_mul_ten(lams[:, 0], x)
    ys = [y]

    if args.pair:
        inside_cnt = 2
    else:
        inside_cnt = (args.klam + 1)//2

    for i in range(1, args.klam):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        if i < inside_cnt:
            # mix private samples
            mixed_x += vec_mul_ten(lams[:, i], x[index, :])
        else:
            # mix public samples
            mixed_x += vec_mul_ten(lams[:, i], x_help[index, :])
        ys.append(y[index])         # Only keep the labels for private samples

    if args.mode == 'instahide':
        sign = torch.randint(2, size=list(x.shape), device=device) * 2.0 - 1
        mixed_x *= sign.float().to(device)
    return mixed_x, ys, lams


def generate_sample(trainloader, inputs_help):
    assert len(trainloader) == 1        # Load all training data once
    for _, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        mix_inputs, mix_targets, lams = mixup_data(
            inputs, targets.float(), inputs_help, use_cuda)
    return (mix_inputs, mix_targets, lams)


def train(net, optimizer, inputs_all, mix_targets_all, lams, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0

    seq = random.sample(range(len(inputs_all)), len(inputs_all))
    bl = list(chunks(seq, args.batch_size))

    for batch_idx in range(len(bl)):
        b = bl[batch_idx]
        inputs = torch.stack([inputs_all[i] for i in b])
        if args.mode == 'instahide' or args.mode == 'mixup':
            lam_batch = torch.stack([lams[i] for i in b])

        mix_targets = []
        for ik in range(args.klam):
            mix_targets.append(
                torch.stack(
                    [mix_targets_all[ik][ib].long().to(device) for ib in b]))
        targets_var = [Variable(mix_targets[ik]) for ik in range(args.klam)]

        inputs = Variable(inputs)
        outputs = net(inputs)
        loss = mixup_criterion(outputs, targets_var, lam_batch)
        train_loss += loss.data.item()
        total += args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(inputs_all)/args.batch_size+1,
                     'Loss: %.3f' % (train_loss / (batch_idx + 1)))
    return (train_loss / batch_idx, 100. * correct / total)


def test(net, optimizer, testloader, epoch, start_epoch):
    global best_acc
    net.eval()
    test_loss, correct_1, correct_5, total = 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)
            total += targets.size(0)
            correct = pred.eq(targets.view(targets.size(0), -
                                           1).expand_as(pred)).float().cpu()
            correct_1 += correct[:, :1].sum()
            correct_5 += correct[:, :5].sum()

            progress_bar(
                batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                    (batch_idx + 1), 100. * correct_1 / total, correct_1, total))

    acc = 100. * correct_1 / total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        save_checkpoint(net, acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss / batch_idx, 100. * correct_1 / total)


def save_checkpoint(net, acc, epoch):
    """ Save checkpoints. """
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    ckptname = os.path.join(
        './checkpoint/', f'{args.model}_{args.data}_{args.mode}_{args.klam}_{args.name}_{args.seed}.t7')
    torch.save(state, ckptname)


def adjust_learning_rate(optimizer, epoch):
    """ Decrease learning rate at certain epochs. """
    lr = args.lr
    if args.data == 'cifar10':
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print('==> Number of lambdas: %g' % args.klam)

    ## --------------- Prepare data --------------- ##
    print('==> Preparing data..')

    cifar_normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_imagenet = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.augment:
        transform_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar_normalize
        ])
    else:
        transform_cifar_train = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize
        ])

    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        cifar_normalize
    ])

    if args.data == 'cifar10':
        trainset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform_cifar_train)
        testset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform_cifar_test)
        trainset_help = datasets.ImageFolder(
            args.help_dir, transform=transform_imagenet)
        num_class = 10
    # You can add your own dataloader and preprocessor here.

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=len(trainset),
                                              shuffle=True,
                                              num_workers=8)

    trainloader_help = torch.utils.data.DataLoader(trainset_help,
                                                   batch_size=len(
                                                       trainset_help),
                                                   shuffle=True,
                                                   num_workers=8)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8)

    ## --------------- Create the model --------------- ##
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.data + '_' +
                                args.name + 'ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[args.model](num_classes=num_class)

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = f'results/log_{args.model}_{args.data}_{args.mode}_{args.klam}_{args.name}_{args.seed}.csv'

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print('==> Using CUDA..')

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=args.decay)

    ## --------------- Train and Eval --------------- ##
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow([
                'Epoch', 'Train loss', 'Test loss',
                'Test acc'
            ])


    for _, (inputs_help, targets) in enumerate(trainloader_help):
        if use_cuda:
            inputs_help = inputs_help.cuda()

    for epoch in range(start_epoch, args.epoch):
        mix_inputs_all, mix_targets_all, lams = generate_sample(
            trainloader, inputs_help)
        train_loss, _ = train(
            net, optimizer, mix_inputs_all, mix_targets_all, lams, epoch)
        test_loss, test_acc1, = test(
            net, optimizer, testloader, epoch, start_epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow(
                [epoch, train_loss, test_loss, test_acc1])


if __name__ == '__main__':
    main()
