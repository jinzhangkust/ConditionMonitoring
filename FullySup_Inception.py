"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
URL: https://github.com/jinzhangkust
Dept: Kunming University of Science and Technology (KUST)
Created on 2026.01.04
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from data import get_stibium_data
from models.inception_proj import inception

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=300, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--resume', type=bool, default=False, help='restore checkpoints')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='FullySupInception')
    # checkpoint
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # create
    opt = parser.parse_args()
    # save
    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class LinearSensor(nn.Module):
    def __init__(self):
        super(LinearSensor, self).__init__()

        self.feature = inception()
        self.predictor = nn.Sequential(
            nn.Dropout(p=0.2),  # add before FC
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x)
        x = self.predictor(x.view(x.size(0), -1))
        return x


def set_loader(opt):
    train_data, val_data, test_data = get_stibium_data()
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def set_model(opt):
    model = LinearSensor()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    if opt.resume:
        resume = os.path.join(opt.save_folder, 'checkpoint_{}.ckpt'.format(opt.epoch))
        print(f"resume from {resume}")
        checkpoint = torch.load(resume, weights_only = False)
        opt.epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer

def cal_metrics(pred, true):
    acc = accuracy_score(pred, true)
    prec = precision_score(pred, true, average='macro')
    recall = recall_score(pred, true, average='macro')
    f1 = f1_score(pred, true, average='macro', zero_division=1)
    return acc, prec, recall, f1

def warmup_learning_rate(opt, epoch, idx, nBatch, optimizer):
    T_total = opt.epochs * nBatch
    T_warmup = 10 * nBatch
    if epoch <= 10 and idx <= T_warmup:
        lr = 1e-6 + (opt.learning_rate - 1e-6) * idx / T_warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, opt, tb):
    model.train()
    meters = AverageMeterSet()
    for idx, (_, images, targets) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]
        # compute loss
        output = model(images)
        loss = criterion(output, targets)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update metric
        _, pred = torch.max(output, 1)
        acc, prec, recall, f1 = cal_metrics(pred.cpu(), targets.cpu())
        meters.update('loss', loss.item(), bsz)
        meters.update('acc', acc, bsz)
        meters.update('prec', prec, bsz)
        meters.update('recall', recall, bsz)
        meters.update('f1', f1, bsz)
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.3f}\t'
                  'Accuracy {acc:.3f}\t'
                  'Precision {prec:.3f}\t'
                  'Recall {recall:.3f}\t'
                  'F1-score {f1:.3f}'.format(
                epoch, idx + 1, len(train_loader), loss=meters['loss'].avg, acc=meters['acc'].avg,
                prec=meters['prec'].avg, recall=meters['recall'].avg, f1=meters['f1'].avg))
            sys.stdout.flush()
    # tensorboard
    tb.add_scalar("Train/Loss", meters['loss'].avg, epoch)
    tb.add_scalar("Train/Accuracy", meters['acc'].avg, epoch)
    tb.add_scalar("Train/Precision", meters['prec'].avg, epoch)
    tb.add_scalar("Train/Recall", meters['recall'].avg, epoch)
    tb.add_scalar("Train/F1-score", meters['f1'].avg, epoch)


def validate(val_loader, model, criterion, epoch, opt, tb):
    model.eval()
    meters = AverageMeterSet()
    with torch.no_grad():
        end = time.time()
        for idx, (_, images, targets) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            bsz = targets.shape[0]
            # forward
            output = model(images)
            loss = criterion(output, targets)
            # update metric
            _, pred = torch.max(output, 1)
            acc, prec, recall, f1 = cal_metrics(pred.cpu(), targets.cpu())
            meters.update('loss', loss.item(), bsz)
            meters.update('acc', acc, bsz)
            meters.update('prec', prec, bsz)
            meters.update('recall', recall, bsz)
            meters.update('f1', f1, bsz)
            # print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'loss {loss:.3f}\t'
                      'Accuracy {acc:.3f}\t'
                      'Precision {prec:.3f}\t'
                      'Recall {recall:.3f}\t'
                      'F1-score {f1:.3f}'.format(
                    idx, len(val_loader), loss=meters['loss'].avg, acc=meters['acc'].avg,
                    prec=meters['prec'].avg, recall=meters['recall'].avg, f1=meters['f1'].avg))
                sys.stdout.flush()
    # tensorboard
    tb.add_scalar("Test/Loss", meters['loss'].avg, epoch)
    tb.add_scalar("Test/Accuracy", meters['acc'].avg, epoch)
    tb.add_scalar("Test/Precision", meters['prec'].avg, epoch)
    tb.add_scalar("Test/Recall", meters['recall'].avg, epoch)
    tb.add_scalar("Test/F1-score", meters['f1'].avg, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="FullySupInception")

    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    for epoch in range(opt.epoch, opt.total_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        train(train_loader, model, criterion, optimizer, epoch, opt, tb)
        time2 = time.time()
        validate(val_loader, model, criterion, epoch, opt, tb)
        if epoch % opt.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, opt.save_folder, epoch)


def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint_{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}
    def __getitem__(self, key):
        return self.meters[key]
    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}
    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}
    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}
    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
