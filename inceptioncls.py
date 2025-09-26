"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
Dept: Kunming University of Science and Technology
Created on 2024.04.18
"""
import random

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.inception_proj import inception

import os
import sys
import time
import argparse
import numpy as np

from data import get_froth_data, data4cls

from util import AverageMeter, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=110, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--total_epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=300, help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # model dataset
    parser.add_argument('--model_name', type=str, default='InceptionClS')
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

class InceptionSensor(nn.Module):
    def __init__(self):
        super(InceptionSensor, self).__init__()
        self.feature = inception()
        self.feature.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512))
        self.classifier = nn.Linear(512, 6)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, x):
        x = self.feature(x).view(x.size(0), -1)
        code = self.projector(x)
        out = self.classifier(code)
        return code, out


def set_model():
    model = InceptionSensor()
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer


def set_loader(opt):
    train_data, val_data, test_data = get_froth_data()
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader


def train(train_loader, model, criterion, optimizer, epoch, tb):
    model.train()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = 0
    end = time.time()
    for idx, (_, (im_w, _), labels) in enumerate(train_loader):
        im_w = im_w.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # compute output
        code_w, out_w = model(im_w)
        # cross entropy loss
        loss = criterion(out_w, labels)
        # update metric
        total_loss += loss.item()
        acc = accuracy(out_w, labels)
        top1.update(acc[0], labels.size(0))
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # tensorboard
    tb.add_scalar("InceptionClSTrain/Acc", top1.avg, epoch)
    tb.add_scalar("InceptionClSTrain/Loss", total_loss, epoch)


def val(val_loader, model, criterion, epoch, tb):
    model.eval()
    top1 = AverageMeter()
    total_loss = 0
    with torch.no_grad():
        for idx, (_, (im_w, _), labels) in enumerate(val_loader):
            im_w = im_w.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # compute loss
            code_w, out_w = model(im_w)
            # cross entropy loss
            loss = criterion(out_w, labels)
            # update metric
            acc = accuracy(out_w, labels)
            top1.update(acc[0], labels.size(0))
            total_loss += loss.item()
    # tensorboard
    tb.add_scalar("InceptionClSVal/Acc", top1.avg, epoch)
    tb.add_scalar("InceptionClSVal/Loss", total_loss, epoch)


def main():
    best_acc = 0
    opt = parse_option()
    tb = SummaryWriter(comment="InceptionClS")
    train_loader, val_loader, test_loader = set_loader(opt)
    model, criterion = set_model()
    optimizer = set_optimizer(opt, model)
    no_pretrain = True
    if opt.epoch > 1:
        load_file = os.path.join(opt.save_folder, 'checkpoint_{epoch}.pth'.format(epoch=opt.epoch))
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model'])
    for epoch in range(opt.epoch + 1, 301):
        # adjust_learning_rate(opt, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, tb)
        val(val_loader, model, criterion, epoch, tb)
        if epoch % opt.save_freq == 0 and epoch >= 1:
            save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
            }, opt.save_folder, epoch)

def save_checkpoint(state, save_folder, epoch):
    filename = os.path.join(save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
    torch.save(state, filename)


if __name__ == '__main__':
    main()
