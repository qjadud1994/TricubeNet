import argparse
import cv2
import numpy as np
import os
import torch
import time
import sys

sys.path.append('..')

from basenet.model import Model_factory
from loader import ListDataset
from utils.post_processing import get_center_point_contour
from utils.post_processing import _nms, _topk, _smoothing, post_processing_for_orientedbox
from utils.util import COLORS, DOTA_CLASSES, tensor_rotate


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='DB/')
parser.add_argument('--checkpoint', type=str, help='trained checkpoint')
parser.add_argument('--experiment', default=1, type=int, help='number of experiment')
parser.add_argument('--input_size', default=1024, type=int, help='input size')
parser.add_argument('--c_thresh', default=0.1, type=float, help='threshold for center point')
parser.add_argument('--s_thresh', default=0.4, type=float, help='threshold for width and height')
parser.add_argument('--scale', default=2.2, type=float, help='scale factor')
parser.add_argument('--kernel', default=3, type=int, help='kernel of max-pooling for center point')
parser.add_argument('--backbone', type=str, default='DLA_dcn', help='[hourglass104, DLA_dcn, uesnet101_dcn]')
parser.add_argument('--mode', default='test', type=str,  help='mode')
parser.add_argument('--flip', type=str2bool, default=False, help='flip augmentation')
parser.add_argument('--ma', type=str2bool, default=False, help='multi-angle augmentation')
parser.add_argument('--save_img', type=str2bool, default=False, help='save result images')

opt = parser.parse_args()
print(opt)

result_img_path = "result_dota_imgs/"
if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)
os.system("rm -rf result_dota_imgs/*.jpg")

""" data loader """
out_size = int(opt.input_size / 2)
mean = (0.485,0.456,0.406)
var = (0.229,0.224,0.225)

testset = ListDataset(root=opt.root, dataset='DOTA', mode=opt.mode, transform=None, 
                       out_size=out_size, evaluation=True)


""" Networks """
num_classes = 15
model = Model_factory(opt.backbone, num_classes)


""" set CUDA """
model.cuda()

checkpoint = torch.load('../checkpoints/%s.pth' % opt.checkpoint)

model.load_state_dict(checkpoint['model'])
checkpoint = None

""" evaluation mode  """
_ = model.eval()

det_dir = '/data/DB/DOTA/results/%s_%d/Task1/' % (opt.checkpoint, opt.experiment)
if not os.path.exists(det_dir):
    os.makedirs(det_dir)
filedict = {}

for cls in DOTA_CLASSES:
    fd = open(os.path.join(det_dir, 'Task1_') + cls + r'.txt', 'a')
    filedict[cls] = fd
    
obj_write = "%s %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" #imgname, conf, box*8


sum_total_time = 0
sum_infer_time = 0
sum_post_time = 0

num = 0

for idx in range(len(testset)):
    img, img_path, target = testset.__getitem__(idx)
    
    img_name = img_path.split("/")[-1][:-4]
    
    ori_h, ori_w, _ = img.shape
    
    h, w = opt.input_size, opt.input_size #512, 512
    img = cv2.resize(img, (w, h))
    
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

        if opt.flip and opt.ma:
            x0   = tensor_rotate(x, 0)
            x90  = tensor_rotate(x, 90)
            x180 = tensor_rotate(x, 180)
            x270 = tensor_rotate(x, 270)
            
            x0_flip   = torch.flip(x0, [3])
            x90_flip  = torch.flip(x90, [3])
            x180_flip = torch.flip(x180, [3])
            x270_flip = torch.flip(x270, [3])
            
            x = torch.cat( [x0, x90, x180, x270, 
                            x0_flip, x90_flip, x180_flip, x270_flip], dim=0)

            out = model(x)

            x0, x90, x180, x270 = None, None, None, None
            x0_flip, x90_flip, x180_flip, x270_flip = None, None, None, None
            
            if 'hourglass' in opt.backbone:
                out = out[1]

            out0        = out[0:1]
            out90       = out[1:2]
            out180      = out[2:3]
            out270      = out[3:4]
            out0_flip   = out[4:5]
            out90_flip  = out[5:6]
            out180_flip = out[6:7]
            out270_flip = out[7:8]
            
            out0   = tensor_rotate(out0, 0, transpose=True)
            out90  = tensor_rotate(out90, 270, transpose=True)
            out180 = tensor_rotate(out180, 180, transpose=True)
            out270 = tensor_rotate(out270, 90, transpose=True)

            out0_flip   = torch.flip(out0_flip, [2])
            out90_flip  = torch.flip(out90_flip, [2])
            out180_flip = torch.flip(out180_flip, [2])
            out270_flip = torch.flip(out270_flip, [2])
            
            out0_flip   = tensor_rotate(out0_flip, 0, transpose=True)
            out90_flip  = tensor_rotate(out90_flip, 270, transpose=True)
            out180_flip = tensor_rotate(out180_flip, 180, transpose=True)
            out270_flip = tensor_rotate(out270_flip, 90, transpose=True)
            
            out = (out0 + out90 + out180 + out270 + 
                   out0_flip + out90_flip + out180_flip + out270_flip) / 8.

            out0, out90, out180, out270 = None, None, None, None
            out0_flip, out90_flip, out180_flip, out270_flip = None, None, None, None
        
        
        elif opt.flip:
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
            
            
        elif opt.ma:
            x90 = tensor_rotate(x, 90)
            x180 = tensor_rotate(x, 180)
            x270 = tensor_rotate(x, 270)
            
            x = torch.cat( [x, x90, x180, x270] , dim=0)

            out = model(x)

            if 'hourglass' in opt.backbone:
                out = out[1]

            out_90 = out[1:2]
            out_180 = out[2:3]
            out_270 = out[3:4]
            out = out[0:1]
            
            out_90 = tensor_rotate(out_90, 270, transpose=True)
            out_180 = tensor_rotate(out_180, 180, transpose=True)
            out_270 = tensor_rotate(out_270, 90, transpose=True)

            out = (out + out_90 + out_180 + out_270) / 4.

            x90, x180, x270 = None, None, None
            out_90, out_180, out_270 = None, None, None
            
            
        else:
            out = model(x)
            
            if 'hourglass' in opt.backbone:
                out = out[1]

        out = _smoothing(out, opt.kernel)
        peak = _nms(out, kernel=opt.kernel)
        c_ys, c_xs = _topk(peak, K=2000)
                
            
    x = x[0].cpu().detach().numpy()
    out = out[0].cpu().detach().numpy()
    c_xs = c_xs[0].int().cpu().detach().numpy()
    c_ys = c_ys[0].int().cpu().detach().numpy()
    
    x = x.transpose(1, 2, 0)
    x *= var
    x += mean
    x *= 255
    x = x.clip(0, 255).astype(np.uint8)
    
    
    torch.cuda.synchronize()
    t1 = time.time()
        
    results = get_center_point_contour(out, opt.c_thresh, opt.scale, (ori_w, ori_h))
    #results = post_processing_for_orientedbox(out, c_xs, c_ys, opt.c_thresh, opt.s_thresh, opt.scale, (ori_w, ori_h))

    torch.cuda.synchronize()
    t2 = time.time()

    for result in results:
        
        filedict[ DOTA_CLASSES[result['label']] ].write(obj_write % (img_name, result['conf'], 
                             result['rbox'][0][0], result['rbox'][0][1],
                             result['rbox'][1][0], result['rbox'][1][1],
                             result['rbox'][2][0], result['rbox'][2][1],
                             result['rbox'][3][0], result['rbox'][3][1]
                            ))

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
            color = COLORS[label]
            
            target_wh = np.array([[w/ori_w, h/ori_h]], dtype=np.float32)
            box = box * np.tile(target_wh, (4,1))
            
            _img = cv2.drawContours(_img, [box.astype(np.int0)], -1, color, 2) # green

            
        merge_out = np.max(out, axis=-1)
        merge_out = np.clip(merge_out * 255, 0, 255)
        
        binary = (merge_out > opt.s_thresh*255) * 255

        merge_out = cv2.applyColorMap(merge_out.astype(np.uint8), cv2.COLORMAP_JET)
        binary = cv2.applyColorMap(binary.astype(np.uint8), cv2.COLORMAP_JET)
        
        merge_out = cv2.resize(merge_out, (w, h))
        binary = cv2.resize(binary, (w, h))
        
        #merge_out = cv2.addWeighted(merge_out, 0.6, img, 0.4, 0) 
        
        result_img = cv2.hconcat([_img[:, :, ::-1], merge_out, binary])

        cv2.imwrite("%s/%s.jpg" % (result_img_path, img_name), result_img)
    
    
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("AVG TOTAL FPS = %.4f" % (1 / (sum_total_time / len(testset))))
print("AVG INFER FPS = %.4f" % (1 / (sum_infer_time / len(testset))))
print("AVG POST FPS = %.4f" % (1 / (sum_post_time / len(testset))))

print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
