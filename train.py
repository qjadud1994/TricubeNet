import os, time, argparse
import cv2
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.utils.data

from basenet.model import Model_factory
from loader import ListDataset
from loss import SWM_FPEM_Loss 
from utils.lr_scheduler import WarmupPolyLR
from utils.augmentations import Augmentation, Augmentation_test
    
cudnn.benchmark = True

def parse():

    """ set argments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/DB/')
    parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
    parser.add_argument('--input_size', type=int, default=1024, help='input size')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--backbone', type=str, default='hourglass104_MRCB_cascade', 
                        help='[hourglass104_MRCB_cascade, hourglass104_MRCB, hhrnet48, DLA_dcn, uesnet101_dcn]')
    parser.add_argument('--dataset', type=str, default='DOTA', help='training dataset')
    parser.add_argument('--epochs', type=int, default=120, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--print_freq', default=100, type=int, help='interval of showing training conditions')
    parser.add_argument('--train_iter', default=0, type=int, help='number of total iterations for training')
    parser.add_argument('--curr_iter', default=0, type=int, help='current iteration')
    parser.add_argument('--save_path', type=str, default='./weight', help='Model save path')
    parser.add_argument('--resume', default=None, type=str,  help='training restore')
    parser.add_argument('--data_split', default='1024_single', type=str,  help='data split for DOTA')
    parser.add_argument('--alpha', type=float, default=10, help='weight for positive loss, default=10')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for learninf rate decay')

    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='half precision')
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse()

    # fixed seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if type(args.input_size) == int:
        args.input_size = (args.input_size, args.input_size)

    out_size = (args.input_size[0]//2, args.input_size[1]//2)

    mean=(0.485,0.456,0.406)
    var=(0.229,0.224,0.225)

    """ initial parameters for training """
    NUM_CLASSES = {'DOTA' : 18, 'HRSC2016' : 1}
    num_classes = NUM_CLASSES[args.dataset]
    
    """ cuda & distributed """
    ngpus = torch.cuda.device_count()
    if args.local_rank==0: print("ngpus : ", ngpus)
    distributed = ngpus > 1
    device = torch.device('cuda:{}'.format(args.local_rank))
    
    if distributed:
        #os.environ['MASTER_ADDR'] = '127.0.0.1'
        #os.environ['MASTER_PORT'] = '99999'
    
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", 
            rank=args.local_rank, world_size=torch.cuda.device_count()
        )
    
    model = Model_factory(args.backbone, num_classes)
    
    if distributed and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0: print("using synced BN")
        
    torch.cuda.set_device(device)
    model = model.to(device)
    
    # Scale learning rate based on global batch size
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    if torch.distributed.get_world_size() > 1:
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank,
        )
        
    # define loss function (criterion) and optimizer
    criterion = SWM_FPEM_Loss(num_classes=num_classes, alpha=args.alpha)
        
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():

            if os.path.isfile(args.resume):
                if args.local_rank == 0: print("=> loading checkpoint '{}'".format(args.resume))
                state = torch.load(args.resume, map_location='cpu')
                
                #model.module.load_state_dict(state['state_dict'], strict=True)
                model.load_state_dict(state['model'], strict=True)
                #optimizer.load_state_dict(state["optimizer"])
                
                if args.local_rank == 0: print("=> loaded checkpoint ", args.resume)
            else:
                if args.local_rank == 0: print("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    

    """ Get data loader """
    transform_train = Augmentation(args.input_size, mean, var)
    transform_valid = Augmentation_test(args.input_size, mean, var)

    train_dataset = ListDataset(root=args.root, dataset=args.dataset, mode='train', split=args.data_split, 
                           transform=transform_train, out_size=out_size)
    valid_dataset = ListDataset(root=args.root, dataset=args.dataset, mode='val', split=args.data_split, 
                           transform=transform_valid, out_size=out_size)

    if args.local_rank == 0: print("number of train = %d / valid = %d" % (len(train_dataset), len(valid_dataset)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler, drop_last=False)
    
    """ lr scheduler """
    args.train_iter = len(train_loader) * args.epochs
    
    scheduler = WarmupPolyLR(
        optimizer,
        args.train_iter,
        warmup_iters=1000,
        power=0.90
    )
        
    if args.local_rank == 0: print(args)
    
    best_loss = 999999999
    best_dist = 999999999
    start = time.time()
    
    for epoch in range(0, args.epochs):
        if distributed: train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args)

        # evaluate on validation set
        val_loss, val_dist = validate(valid_loader, model, criterion, device, epoch, args)

        # save checkpoint
        if args.local_rank == 0:

            if best_loss <= val_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, "best_loss", args.save_path)

            if best_dist <= val_dist:
                best_dist = val_dist
                save_checkpoint(model, optimizer, epoch, "best_dist", args.save_path)
                    
    
def train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

     # switch to train mode
    model.train()
    end = time.time()
    
    world_size = torch.distributed.get_world_size()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for x, y, w, s in train_loader:
        args.curr_iter += 1
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outs = model(x)

            if type(outs) == list:
                loss = 0
                for out in outs:
                    loss += criterion(y, out, w, s)
                    
                loss /= len(outs)
                    
                outs = outs[-1]

            else:
                loss = criterion(y, outs, w, s)
        
        # compute gradient and backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        reduced_loss = reduce_tensor(loss.data, world_size)
        losses.update(reduced_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.local_rank == 0 and args.curr_iter % args.print_freq == 0:
            train_log = "Epoch: [%d/%d][%d/%d] " % (epoch, args.epochs, args.curr_iter, args.train_iter)
            train_log += "({0:.1f}%, {1:.1f} min) | ".format(args.curr_iter/args.train_iter*100, (end-start) / 60)
            train_log += "Time %.1f ms | Left %.1f min | " % (batch_time.avg * 1000, (args.train_iter - args.curr_iter) * batch_time.avg / 60)
            train_log += "Loss %.6f " % (losses.avg)
            print(train_log)

                
    
def validate(valid_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    distances = AverageMeter()

    # switch to evaluate mode
    model.eval()

    world_size = torch.distributed.get_world_size()
    end = time.time()

    for x, y, w, s in valid_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)

        # compute output
        with torch.no_grad():
            outs = model(x)
            
            if type(outs) == list:
                outs = outs[-1]

            loss = criterion(y, outs, w, s)

        # measure accuracy and record loss
        dist = torch.sqrt((y - outs)**2).mean()

        reduced_loss = reduce_tensor(loss.data, world_size)
        reduced_dist = reduce_tensor(dist.data, world_size)

        losses.update(reduced_loss.item())
        distances.update(reduced_dist.item())

    if args.local_rank == 0:
        valid_log = "\n============== validation ==============\n"
        valid_log += "valid time : %.1f s | " % (time.time() - end)
        valid_log += "valid loss : %.6f | " % (losses.avg)
        valid_log += "valid dist : %.6f \n" % (distances.avg)
        print(valid_log)
        
    return losses.avg, distances.avg


def save_checkpoint(model, optimizer, epoch, name, save_path):
    state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
    model_file = os.path.join(save_path,  f"{name}.pt")
    torch.save(state, model_file)


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


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


if __name__ == '__main__':
    
    main()
    

