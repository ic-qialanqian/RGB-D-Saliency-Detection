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
from DANet_Ours import RGBD_sal
from torch.backends import cudnn
import torch.nn.functional as functional
import argparse
from wavelet_transform.DWT_IDWT_layer import DWT_2D, IDWT_2D
from contrastloss import ContrastiveCenterLoss
import numpy as np
from scipy import misc

class MyContrastiveCenterLoss(nn.Module):  # modified from center loss, we add the term from contrastive center loss, but change it to margin-d
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=1, feat_dim=4096, use_gpu=True):
        super(MyContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss1 = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size  #original center loss

        #add contrastive loss
        #batch_size = x.size()[0]  # for example batch_size=6
        expanded_centers = self.centers.expand(batch_size, -1, -1)  # shape 6,3,2
        expanded_hidden = x.expand(self.num_classes, -1, -1).transpose(1, 0)  # shape 6,3,2
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)  # shape 6,3
        #distances_same = distance_centers.gather(1, labels.unsqueeze(1))  # 6,1
        intra_distances = loss1#distances_same.sum()  # means inner ,distance in the same class
        inter_distances = distance_centers.sum().sub(loss1*batch_size)  # distance between different class ,sub =minus
        inter_distances=inter_distances/batch_size/self.num_classes
        epsilon = 1e-6
        #
        loss2=np.max([margin-inter_distances,0])
        loss=loss1+0.1*loss2

        return loss



to_pil = transforms.ToPILImage()
def visualize_prediction1(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal1.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)




def Dilation(input):
    maxpool = nn.AdaptiveAvgPool2d(5)
    map_b = maxpool(input)
    return map_b
    
def cross_entropy2d(input, target, temperature=1, weight=None, size_average=True):
    target = target.long()
    n, c, h, w = input.size()
    #print(input.size())
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    
    T = temperature
    loss = functional.cross_entropy(input / T, target, weight=weight, size_average=size_average)
    # if size_average:
    #     loss /= mask.data.sum()
    return loss
    
    
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
parser.add_argument('--exp_name', type=str, default='SOTA_baseline_addcd1-5')

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
    #model.load_state_dict(torch.load('/media/guangyu/csp1/projects/Salient-Detection/RGBD-SOD1/RGBD-SOD/model/FPN_baseline/20500.pth'))
    #model.Resnet.load_state_dict(torch.load('/media/guangyu/csp1/projects/PoolNet1/dataset/pretrained/resnet50_caffe.pth'))
    #model.Depth.load_state_dict(torch.load('/media/guangyu/csp1/projects/PoolNet1/dataset/pretrained/resnet50_caffe.pth'))
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
    criterion_con = MyContrastiveCenterLoss(1,256)
    #criterion_con = ContrastiveCenterLoss(1,1).cuda()
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
            
            
            dwt_layer = DWT_2D(wavename='haar')
            LL_gt, LH_gt, HL_gt, HH_gt = dwt_layer(labels)
            high_labels = LH_gt+ HL_gt+HH_gt
            # network
            outputs,c5_overall_ouput,f1_attention,f2_attention,f3_attention,f4_attention,d2 =  net(inputs, depth)
            #outputs,c5_overall_ouput,f1_attention,f2_attention,f3_attention,f4_attention,depth2,masked_depth2 =  net(inputs, depth)
            #outputs =  net(inputs, depth)
            ##########loss#############
            optimizer.zero_grad()
            '''
            loss_depth5 = criterion(depth5_attention, labels)
            loss_depth4 = criterion(depth4_attention, labels)
            loss_depth3 = criterion(depth3_attention, labels)
            loss_depth2 = criterion(depth2_attention, labels)
            '''
            
            
            loss_f0 = criterion(c5_overall_ouput, labels0)
            loss_f1 = criterion(f1_attention, labels2)
            loss_f2 = criterion(f2_attention, labels3)
            loss_f3 = criterion(f3_attention, labels4)
            loss_f4 = criterion(f4_attention, labels)
            
            #loss_d1 = criterion(depth1, labels0)
            #loss_d2 = criterion(depth2, labels4)
            #loss_d3 = criterion(depth3, labels3)
            #print(labels.size())
            #print(outputs.size())
            #loss_high = criterion(high,high_labels)
            #print(loss_contrast)
            #loss = criterion(outputs, labels)
            
            loss = criterion(outputs, labels)
            total_loss = loss+loss_f2+loss_f3+loss_f4+loss_f0+loss_f1


            visualize_prediction1(d2)
            #total_loss = loss
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
