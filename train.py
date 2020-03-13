import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable, DoubleTable, SingleTable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import torch.distributed as dist
import evaluation
import torchvision
from model.mb_tiny_RFB import Mb_Tiny_RFB
from model.centernet import EfficientNet, efficientnet_b0
from model.losses import CtdetLoss
import os
from torch.utils.data.distributed import DistributedSampler
from torch.optim.optimizer import Optimizer, required
import math
from datasets import LoadImagesAndLabels
from utils.utils import RAdam
from collections import OrderedDict
import itertools
from torchsummary import summary

def get_args():
    parser = argparse.ArgumentParser(description="Train program for model.")
    parser.add_argument('--data_path', type=str, default='../../data/data_wider', help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=50, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=5, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=1, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./weights', help='Model save path')
    args = parser.parse_args()
    return args

def main(resume = 0, rfb = False):
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_dir=log_path)

    dataset_train =  LoadImagesAndLabels(root = "/media/hdd/sources/Person_2_heatmap/person", path = "/media/hdd/sources/Person_2_heatmap/person/data/full_path.train", img_size=[512, 512])
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch, shuffle=args.shuffle)

    dataset_val =  LoadImagesAndLabels(root = "/media/hdd/sources/Person_2_heatmap/person", path = "/media/hdd/sources/Person_2_heatmap/person/data/WiderPerson.val", img_size=[512, 512], split = 'val')
    dataloader_val = DataLoader(dataset_val, num_workers=2, batch_size=1, shuffle=False)
    total_batch = len(dataloader_train)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if rfb:
        model = Mb_Tiny_RFB()
    else:
        model = efficientnet_b0(pretrained=False)

    model = model.to(device)
    # summary(model, (3, 640, 640))
    if resume:
        state_dict = torch.load('weights/model_epoch_%s.pt'%resume)
        # create new OrderedDict that does not contain `module.`
        model_dict = model.state_dict()
        lst_key = model_dict.keys()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in lst_key:
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(state_dict)
        del state_dict

    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.0005)
    if resume:
        state_dict = torch.load('out/optimizer_epoch_%s.pt'%resume)
        optimizer.load_state_dict(state_dict)
        del state_dict

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [5, 35,  70], gamma=0.1)
    print('Start to train.')
    epoch_loss = []
    iteration = 0
    model.train()

    loss_fnc = CtdetLoss(device)
    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        if resume:
            if epoch < resume:
                continue

        # # Training
        for iter_num, data in enumerate(dataloader_train):
            # break
            if cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            optimizer.zero_grad()
            output = model(data['input'].float())
            loss, loss_sta = loss_fnc(output, data)
            loss.backward()
            optimizer.step()
            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                table_data = [
                    ['loss name','value'],
                    ['total_loss',str(loss.item())],
                    ['hm_loss', str(loss_sta['hm_loss'].item())],
                    ['wh_loss', str(loss_sta['wh_loss'].item())],
                    ['off_loss', str(loss_sta['off_loss'].item())],
                    ]
                table = AsciiTable(table_data)
                log_str +=table.table
                print(log_str)
                # write the log to tensorboard
                writer.add_scalar('losses:',loss.item(),iteration*args.verbose)
                writer.add_scalar('hm_loss:', loss_sta['hm_loss'].item(),iteration*args.verbose)
                writer.add_scalar('wh loss:',loss_sta['wh_loss'].item(),iteration*args.verbose)
                writer.add_scalar('off loss:',loss_sta['off_loss'].item(),iteration*args.verbose)
                iteration +=1
        if epoch % args.eval_step == 0:
            print('-------- centerface Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision = evaluation.evaluate(dataloader_val, model, cuda = True, resize = (512, 512), threshold=0.35)
            print('Recall:',recall)
            print('Precision:',precision)

            writer.add_scalar('Recall:', recall, epoch*args.eval_step)
            writer.add_scalar('Precision:', precision, epoch*args.eval_step)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(optimizer.state_dict(), args.save_path + '/optimizer_epoch_{}.pt'.format(epoch + 1))
            torch.save(model.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()


if __name__=='__main__':
    main()