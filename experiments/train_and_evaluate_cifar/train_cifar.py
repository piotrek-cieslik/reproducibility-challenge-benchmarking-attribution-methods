# code based on https://github.com/visinf/idsds/blob/main/train.py

import argparse
import time
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from  torch.nn.modules.upsampling import Upsample

from utils.log import AverageMeter, ProgressMeter, Summary, accuracy
from utils.cifar_utils import get_cifar100_loaders, GaussianSmoothing


parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--step_size', default=10, type=int,
                    metavar='N', help='step size of learning rate scheduler')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')
parser.add_argument('--pretrained_ckpt_path', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. If None, all GPUs are used')
parser.add_argument('--number_classes', default=100, type=int,
                    help='number of classes')

parser.add_argument('--grid_rows_and_cols', default=4, type=int,
                    help='number of rows and cols in the intervention grid')
parser.add_argument('--baseline', required=False, default='zeros',
                    choices=['zeros', 'blur', 'random'],
                    help='baseline for perturbation')

parser.add_argument('--store_path', default='./', type=str, metavar='PATH',
                    help='path to store the checkpoints')


import os
def save_checkpoint(state, is_best, store_path, model_name):
    filename = os.path.join(store_path, model_name + '_checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(store_path, model_name + '_checkpoint_best.pth.tar')
        torch.save(state, filename_best)

def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpu:
        device = 'cuda:' + str(args.gpu)
    else:
        device = 'cuda'

    train_loader, val_loader = get_cifar100_loaders(args)

    for model_name in ['cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44', 'cifar100_resnet56', 'cifar100_vgg11_bn', 'cifar100_vgg13_bn', 'cifar100_vgg16_bn', 'cifar100_vgg19_bn']:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=False)

        model_path = args.pretrained_ckpt_path + '/' + model_name + '.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion), optimizer, and learning rate scheduler
        ce_criterion = nn.CrossEntropyLoss().to(device)


        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

        best_acc1 = 0

        if args.evaluate:
            validate(args, val_loader, model, ce_criterion, args, device)
            return

        for epoch in range(0, args.epochs):

            # train for one epoch
            train(args, train_loader, model, ce_criterion, optimizer, epoch, device)

            # evaluate on validation set
            acc1 = validate(args, val_loader, model, ce_criterion, device)
            scheduler.step()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'args' : args
            }, is_best, args.store_path, model_name)



def train(args, train_loader, model, ce_criterion, optimizer, epoch, device):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    scale = Upsample(size=(32,32), mode='nearest')

    blur = GaussianSmoothing(3, kernel_size=7, sigma=5, device=device)

    for i, (images, targets, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        images.requires_grad = True
        B,C,H,W = images.shape
        targets = targets.to(device, non_blocking=True)


        # randomly delete patches to make interventions in-domain
        if args.baseline == 'zeros':
            baseline = torch.zeros_like(images)
        elif args.baseline == 'random':
            baseline = torch.rand_like(images) * 2. - 1.
        elif args.baseline == 'blur':
            baseline = blur(images.clone())
        else:
            print('baseline not implemented')

        delete_patches = torch.ones((B,1,args.grid_rows_and_cols,args.grid_rows_and_cols)).float().to(device) # array of 1
        rand_rows = torch.randint(0, args.grid_rows_and_cols, (B,))
        rand_cols = torch.randint(0, args.grid_rows_and_cols, (B,))
        batch_indices = torch.arange(B)
        delete_patches[batch_indices,:,rand_rows,rand_cols] = 0.
        interventions_for_sample = torch.randint(0, 2, (B,)).to(device) # 50% of the images should not have any interventions, so we create another delete patch along the batch dimension that defines if an image has interventions or not
        delete_patches[torch.nonzero(interventions_for_sample), :, :, :] = 1.
        delete_patches = scale(delete_patches)
        delete_patches_inverse = (delete_patches == 0).float() # for baseline
        
        images = images * delete_patches
        baseline = baseline * delete_patches_inverse

        images = images + baseline

        output = model(images)

        loss = ce_criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(args, val_loader, model, criterion, device):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target, _) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(device, non_blocking=True)
                target = target.cuda(device, non_blocking=True)

                # compute output
                output = model(images)
                #loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                #losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg
    
if __name__ == '__main__':
    main()
