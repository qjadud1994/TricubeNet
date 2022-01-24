import torch
import numpy as np
import cv2


def get_center_point_contour(output, thresh, scale, ori_size):
    ori_w, ori_h = ori_size
    
    height, width, num_classes = output.shape

    c_mask = (output > thresh).astype(np.uint8)

    results = []
    
    for cls in range(num_classes):
        mask = c_mask[:, :, cls]

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for k in range(1,nLabels):
            size = stats[k, cv2.CC_STAT_AREA]

            # make segmentation map
            segmap = np.zeros_like(mask, dtype=np.uint8)
            segmap[labels==k] = 255
            #cv2.dilate(segmap, kernel, segmap)

            im2, contours, hierarchy = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for cnt in contours:
                # compute the center of the contour
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                rect = cv2.minAreaRect(cnt)
                rect = ((rect[0][0]*ori_w/width, rect[0][1]*ori_h/height), 
                        (rect[1][0]*scale*ori_w/width, rect[1][1]*scale*ori_h/height), rect[2])
                
                box = cv2.boxPoints(rect)
                
                results.append({"conf" : max(0.0 ,min(1.0, output[cy, cx, cls])),
                               "rbox" : box, 
                               "label" : cls})    
    
    return results


def get_center_point_contour_HRSC(output, thresh, scale, ori_size):
    ori_w, ori_h = ori_size
    
    height, width, num_classes = output.shape

    c_mask = (output > thresh).astype(np.uint8)

    results = []
    
    for cls in range(num_classes):
        mask = c_mask[:, :, cls]

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for k in range(1,nLabels):
            size = stats[k, cv2.CC_STAT_AREA]

            # make segmentation map
            segmap = np.zeros_like(mask, dtype=np.uint8)
            segmap[labels==k] = 255
            #cv2.dilate(segmap, kernel, segmap)

            im2, contours, hierarchy = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for cnt in contours:

                # compute the center of the contour
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                bbox = cv2.boundingRect(cnt)
                bbox = (bbox[0]*ori_w/width, bbox[1]*ori_h/height, 
                        bbox[2]*ori_w/width*scale, bbox[3]*ori_h/height*scale)
                
                bbox = (bbox[0], bbox[1], 
                        bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
                
                rect = cv2.minAreaRect(cnt)
                
                (rbox_x, rbox_y), (rbox_width, rbox_height), rect_angle = rect

                if rbox_width < rbox_height:
                    rbox_width, rbox_height = rbox_height, rbox_width
                    rect_angle += 90
                
                rect = ((rbox_x*ori_w/width, rbox_y*ori_h/height), 
                        (rbox_width*ori_w/width*scale, rbox_height*ori_h/height*scale), 
                        rect_angle)
                
                box = cv2.boxPoints(rect)

                results.append({"conf" : max(0.0 ,min(1.0, output[cy, cx, cls])),
                               "rbox" : box, "rect" : rect, "bbox" : bbox,
                               "label" : cls})    
    
    return results



def _smoothing(heat, kernel=3):
    pad = (kernel - 1) // 2

    heat = heat.permute(0, 3, 1, 2) #[bs, C, H, W]

    heat = torch.nn.functional.avg_pool2d(heat, (kernel, kernel), stride=1, padding=pad) # smoothing

    heat = heat.permute(0, 2, 3, 1)

    return heat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    heat = heat.clone().permute(0, 3, 1, 2) #[bs, C, H, W]

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()

    peak = heat * keep
    peak = peak.permute(0, 2, 3, 1)

    return peak


def _topk(scores, K=40):
    batch, height, width, cat = scores.size()
    
    scores = scores.permute(0, 3, 1, 2)
    
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    
    return topk_ys, topk_xs



def region_growing(output, seed, conf, s_thresh):
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [seed]

    #Mean of the segmented region
    output = output.copy() / conf

    #Input image parameters
    height, width = output.shape

    #Initialize segmented output image
    mask = np.zeros((height, width), np.uint8)

    while len(stack) > 0:
        seed = stack.pop()
        
        if mask[seed] == 255:
            continue
            
        mask[seed] = 255
    
        for i in range(4):
            y_new = seed[0] + neighbors[i][0]
            x_new = seed[1] + neighbors[i][1]
            
            seed_new = (y_new, x_new)

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < width) & (y_new < height)
            
            if check_inside and output[seed_new] > s_thresh and mask[seed_new] == 0:
                stack.append( seed_new )
    
    return mask

    

def post_processing_for_orientedbox(output, c_xs, c_ys, c_thresh, s_thresh, scale, ori_size):
    ori_w, ori_h = ori_size
    height, width, num_classes = output.shape
    
    results = []
    
    output[output < 0.1] = 0
    
    for cls in range(num_classes):
        c_x = c_xs[cls]
        c_y = c_ys[cls]

        for x, y in zip(c_x, c_y):
            conf = output[y, x, cls]

            if conf < c_thresh:
                 break

            seg_mask = region_growing(output[:, :, cls].copy(), (y, x), conf, s_thresh)
            
            _, contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                rect = ((rect[0][0]*ori_w/width, rect[0][1]*ori_h/height), 
                        (rect[1][0]*scale*ori_w/width, rect[1][1]*scale*ori_h/height), rect[2])

                box = cv2.boxPoints(rect)

                results.append({"conf" : max(0.0 ,min(1.0, conf)),
                               "rbox" : box, 
                               "label" : cls})  
                
    return results
