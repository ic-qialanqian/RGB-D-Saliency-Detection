import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from Mask-Guided-Net import RGBD_sal
from torch.backends import cudnn
import torch.nn.functional as functional
import argparse
import numpy as np
from scipy import misc

    
    
cudnn.benchmark = True
# set seeds
torch.manual_seed(2018)
torch.cuda.set_device(0)
# re-define the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--iter_num', type=int, default=20500)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--last_iter', type=int, default=0)
#parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default='./model')
parser.add_argument('--exp_name', type=str, default='mask-guided-net')

##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(256,256),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
target_transform = transforms.ToTensor()
##########################################################################
def main(args):
    # define models 
    model = RGBD_sal().cuda()
    net = model.train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum)
    if len(args.snapshot) > 0:
        print ('training resumes from ' + args.snapshot)
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    # open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer, args)

def train(net, optimizer, args):
    train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=12, shuffle=True)
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')
    # loss
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_BCE = nn.BCELoss().cuda()
    criterion_MAE = nn.L1Loss().cuda()
    criterion_MSE = nn.MSELoss().cuda()
   
    
    curr_iter = args.last_iter
    # start training
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record = \
                AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / args.iter_num) ** args.lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / args.iter_num) ** args.lr_decay
            inputs, depth, labels= data
            labels[labels>0.5] = 1
            labels[labels!=1] = 0
            batch_size = inputs.size(0)
            # the input variables
            inputs = Variable(inputs).cuda()
            depth = Variable(depth).cuda()
            labels = Variable(labels).cuda()
            
            labels0 = functional.interpolate(labels, size=16, mode='bilinear')
            labels1 = functional.interpolate(labels, size=17, mode='bilinear')
            labels2 = functional.interpolate(labels, size=32, mode='bilinear')
            labels3 = functional.interpolate(labels, size=64, mode='bilinear')
            labels4 = functional.interpolate(labels, size=128, mode='bilinear')
            
            # network
            outputs,c5_overall_ouput,f1_attention,f2_attention,f3_attention,f4_attention =  net(inputs, depth)
            
            ##########loss#############
            optimizer.zero_grad()
            
            
            
            loss_f0 = criterion(c5_overall_ouput, labels0)
            loss_f1 = criterion(f1_attention, labels2)
            loss_f2 = criterion(f2_attention, labels3)
            loss_f3 = criterion(f3_attention, labels4)
            loss_f4 = criterion(f4_attention, labels)
            
           
            
            loss = criterion(outputs, labels)
            total_loss = loss+loss_f0+loss_f1+loss_f2+loss_f3+loss_f4


           
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            
            loss1_record.update(loss_f0.item(), batch_size)
            loss2_record.update(loss_f1.item(), batch_size)
            loss3_record.update(loss_f2.item(), batch_size)
            loss4_record.update(loss_f3.item(), batch_size)
            loss5_record.update(loss_f4.item(), batch_size)
            loss6_record.update(loss.item(), batch_size)
            #loss7_record.update(loss_c5.item(), batch_size)
            
            curr_iter += 1
            #############log###############
            if curr_iter % 20500 == 0 or curr_iter % 16400 == 0:
                torch.save(net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, '{}.pth'.format(curr_iter)))
                torch.save(optimizer.state_dict(), os.path.join(args.ckpt_path, args.exp_name, '{}_optim.pth'.format(curr_iter)))
            log = '[iter {}], [total loss {:.5f}],[loss_f4 {:.5f}] '.format(curr_iter, total_loss,loss_f4)
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args.iter_num:
                torch.save(net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, '{}.pth'.format(curr_iter)))
                torch.save(optimizer.state_dict(), os.path.join(args.ckpt_path, args.exp_name, '{}_optim.pth'.format(curr_iter)))
                return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
