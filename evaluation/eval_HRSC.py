import torch
#torch.backends.cudnn.benchmark = True

import argparse
import cv2
import numpy as np
import os
import time
import sys

sys.path.append('..')

from basenet.model import Model_factory
from loader import ListDataset
from utils.post_processing import get_center_point_contour_HRSC
from utils.util import COLORS, tensor_rotate
from utils.bbox_util import keep_resize

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/DB/')
parser.add_argument('--checkpoint', type=str, help='select training dataset')
parser.add_argument('--experiment', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--input_size', default=512, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--s_thresh', default=0.12, type=float, help='Number of workers used in dataloading')
parser.add_argument('--c_thresh', default=0.4, type=float, help='Number of workers used in dataloading')
parser.add_argument('--kernel', default=7, type=int, help='Number of workers used in dataloading')
parser.add_argument('--scale', default=1.42, type=float, help='Number of workers used in dataloading')
parser.add_argument('--backbone', type=str, default='hourglass', help='[vgg16, resnet50, se-resnet50, pspnet, pvanet]')
parser.add_argument('--flip', type=str2bool, default=False, help='save result images')
parser.add_argument('--ms', type=str2bool, default=False, help='save result images')
parser.add_argument('--save_img', type=str2bool, default=False, help='save result images')

opt = parser.parse_args()
print(opt)

result_img_path = "result_HRSC_imgs/"
if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)
os.system("rm -rf result_HRSC_imgs/*.jpg")


""" data loader """
opt.input_size = (512, 512)  #h, w
out_size = (opt.input_size[0]//2, opt.input_size[1]//2)

mean = (0.485,0.456,0.406)
var = (0.229,0.224,0.225)

testset = ListDataset(root=opt.root, dataset='HRSC2016', mode='test', transform=None, 
                       out_size=out_size, evaluation=True)


""" Networks """
num_classes = 1
model = Model_factory(opt.backbone, num_classes)


""" set CUDA """
model.cuda()

checkpoint = torch.load('../checkpoints/%s.pth' % opt.checkpoint)

model.load_state_dict(checkpoint['model'])
checkpoint = None

""" evaluation mode  """
_ = model.eval()


obj_write = "%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.6f\n"    


# warm up
for _ in range(200):
    out = model(torch.randn(2 if opt.flip else 1, 3, opt.input_size[0], opt.input_size[1]).cuda())
    if opt.backbone == 'hourglass':
        out = out[1]
    
sum_total_time = 0
sum_infer_time = 0
sum_post_time = 0

num = 0

for idx in range(len(testset)):
    img, img_path, target = testset.__getitem__(idx)
    
    gt_label = [t[4] for t in target]
    gt_label = list(set(gt_label))
    
    img_name = img_path.split("/")[-1].split(".")[0]
    
    ori_h, ori_w, _ = img.shape

    #h, w = opt.input_size
    img, (w, h) = keep_resize(img, opt.input_size, mean)
    #img = cv2.resize(img, (w, h))
    
    x = img.copy()
    x = x.astype(np.float32)
    x /= 255.
    x -= mean
    x /= var
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
    x = x.unsqueeze(0)
    
    with torch.no_grad():

        x = x.to('cuda')

        torch.cuda.synchronize()
        t0 = time.time()

        if opt.flip:
            x_flip = torch.flip(x, [3])
            x = torch.cat( [x, x_flip] , dim=0)

            out = model(x)
            if 'hourglass' in opt.backbone:
                out = out[1]

            out_flip = out[1:2]
            out = out[0:1]

            out_flip = torch.flip(out_flip, [2])
            out = (out + out_flip) / 2

            x_flip, out_flip = None, None
            
        elif opt.ms:
            x180 = tensor_rotate(x, 180)
            
            x_flip = torch.flip(x, [3])
            x180_flip = torch.flip(x180, [3])
            
            x = torch.cat( [x, x_flip, x180, x180_flip] , dim=0)

            out = model(x)

            if 'hourglass' in opt.backbone:
                out = out[1]

            out_flip = out[1:2]
            out_180 = out[2:3]
            out_180_flip = out[3:4]
            out = out[0:1]
            
            out_flip = torch.flip(out_flip, [2])
            out_180 = tensor_rotate(out_180, 180, transpose=True)
            out_180_flip = torch.flip(out_180_flip, [2])
            out_180_flip = tensor_rotate(out_180_flip, 180, transpose=True)
            
            out = (out + out_flip + out_180 + out_180_flip) / 4.

            x_flip, x180, x180_flip = None, None, None
            out_flip, out_180, out_180_flip = None, None, None
            
            
        else:
            out = model(x)
            
            if 'hourglass' in opt.backbone:
                out = out[1]

        torch.cuda.synchronize()
        t1 = time.time()
        
    if True:
        w_s = (opt.input_size[1] - w)//2
        w_e = w_s + w
        h_s = (opt.input_size[0] - h)//2
        h_e = h_s + h

        out = out[:, h_s//2 : h_e//2, w_s//2 : w_e//2, :]
        img = img[h_s : h_e, w_s : w_e, :]

        
    out = out[0].cpu().detach().numpy()
    
    results = get_center_point_contour_HRSC(out, opt.c_thresh, opt.scale, (ori_w, ori_h))


    torch.cuda.synchronize()
    t2 = time.time()

    sum_total_time += (t2 -t0)
    sum_infer_time += (t1 -t0)
    sum_post_time  += (t2 -t1)

    left = (sum_total_time / (idx+1)) * (len(testset) - idx)
    
    print("[%d/%d]  %s  , FPS=%.2f (Infer=%.2f + post=%.2f) ,  left=%.2f" 
          % (idx, len(testset), img_name, 
             1 / (sum_total_time / (idx+1)), 
             1 / (sum_infer_time / (idx+1)), 
             1 / (sum_post_time / (idx+1)), 
             left), end='\r')

    if opt.save_img:
        _img = img.copy()

        for result in results:
            box = result['rbox']
            label = result['label']
            
            target_wh = np.array([[w/ori_w, h/ori_h]], dtype=np.float32)
            box = box * np.tile(target_wh, (4,1))
            
            _img = cv2.drawContours(_img, [box.astype(np.int0)], -1, (0, 255, 0), 2) # green
                
        merge_out = np.max(out, axis=-1)
        merge_out = np.clip(merge_out * 255, 0, 255)

        merge_out = cv2.applyColorMap(merge_out.astype(np.uint8), cv2.COLORMAP_JET)
        merge_out = cv2.resize(merge_out, (w, h))

        #merge_out = cv2.addWeighted(merge_out, 0.6, _x, 0.4, 0) 
        result_img = cv2.hconcat([_img[:, :, ::-1], merge_out])

        cv2.imwrite("%s/%s.jpg" % (result_img_path, img_name), result_img)
    
    
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("AVG TOTAL FPS = %.4f" % (1 / (sum_total_time / len(testset))))
print("AVG INFER FPS = %.4f" % (1 / (sum_infer_time / len(testset))))
print("AVG POST FPS = %.4f" % (1 / (sum_post_time / len(testset))))


print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
