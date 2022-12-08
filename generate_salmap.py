import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import test_data
from misc import check_mkdir, crf_refine
from Mask-Guided-Net import RGBD_sal
import argparse
import cv2
import torch.nn.functional as functional
torch.manual_seed(2022)
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='./model')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--snapshot', type=str, default='20500')
parser.add_argument('--crf_refine', type=bool, default=False)
parser.add_argument('--save_results', type=bool, default=True)

# normalize
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
# 
depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()
to_test = {'test':test_data}

def main(args):
    t0 = time.time()
    net = RGBD_sal().cuda()
    print ('load snapshot \'%s\' for testing' % args.snapshot)
    net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            root1 = os.path.join(root,'GT')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.png')]
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'RGB',img_name + '.jpg')).convert('RGB')                
                depth = Image.open(os.path.join(root,'depth',img_name + '.png')).convert('L')
                gt = Image.open(os.path.join(root,'GT',img_name + '.png')
                
                img = img1
                w_,h_ = img1.size
                img1 = img1.resize([256,256])
                depth = depth.resize([256,256])
                gt = gt.resize([16,16])
                img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                depth = Variable(depth_transform(depth).unsqueeze(0), volatile=True).cuda()
                gt = Variable(depth_transform(gt).unsqueeze(0), volatile=True).cuda()
               
                prediction = net(img_var,depth)
               
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
           
                
                
                if args.crf_refine:
                    prediction = crf_refine(np.array(img), np.array(prediction))
                prediction = np.array(prediction)
                if args.save_results:
                    check_mkdir('{}/{}/masked_depth2'.format(args.ckpt_path, args.exp_name))
                    Image.fromarray(prediction).save(os.path.join('{}/{}/masked_depth2'.format(args.ckpt_path, args.exp_name), img_name + '.png'))
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
