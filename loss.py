import numpy as np
import torch
import torch.nn as nn


class SWM_FPEM_Loss(nn.Module):
    def __init__(self, num_classes, alpha, neg_pos_ratio=3):
        super(SWM_FPEM_Loss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, y, out, w, total_size):
        B, H, W, C = y.size()
        
        y = y.contiguous().view(B, -1)
        out = out.contiguous().view(B, -1)
        w = w.contiguous().view(B, -1)
        total_size = total_size.squeeze(1)
        
        """ positive samples """
        pos_idx = (w > 0).float()
        
        """ false-positive samples """
        neg_idx = ( (out > 0) > (w > 0) ).float()

        mse_loss = torch.pow( out - y, 2)

        """ positive loss & negative loss """
        pos_loss = (w * mse_loss * pos_idx)
        neg_loss = (mse_loss * neg_idx)

        """ negative sampling """
        train_loss = 0
        for b in range(B):
            if total_size[b] > 0:
                sampling = total_size[b].int().item() * self.neg_pos_ratio
                sampling = min(sampling, H*W*C)
                
                _pos_loss = pos_loss[b].sum()
                _neg_loss = neg_loss[b].topk(sampling)[0].sum()

                train_loss += (self.alpha*_pos_loss + _neg_loss) / total_size[b]

        train_loss /= B
        mse_loss = mse_loss.mean()
        
        return (train_loss + mse_loss) * 10    
   
        #return train_loss * 10
    
