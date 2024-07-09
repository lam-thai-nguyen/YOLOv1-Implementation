"""
Last edited on: Jul 9, 2024
by: Lam Thai Nguyen
"""

import torch
import torch.nn as nn
from utils import IoU


class YOLOv1Loss(nn.Module):
    def __init__(self):
        super(YOLOv1Loss, self).__init__()
        self.lambda_coord = 5  # paper hyperparameter
        self.lambda_noobj = 0.5  # paper hyperparameter
        self.mse = nn.MSELoss(reduction="sum")
        
    def forward(self, pred, true):
        # YOLOv1 Loss function has 4 parts
        
        # Calculate IoU from (x, y, w, h)
        box_true = true[..., 21:25]
        IoU_box_1 = IoU(pred[..., 21:25], box_true)
        IoU_box_2 = IoU(pred[..., 26:30], box_true)
        
        # Original paper functions (I_1 and I_2) 
        I_1 = true[..., 20:21]  # I_1 denotes if object appears in cell (I_1 = 0 or I_1 = 1)
        IoU_cat = torch.cat((IoU_box_1, IoU_box_2), dim=0)
        I_2 = torch.argmax(IoU_cat, dim=0, keepdim=True)  # I_2 denotes that the argmax^th object is responsible for that prediction (I_2 = 0 or I_2 = 1)
        
        # Mask for the responsible box
        box1_mask = (I_2 == 0).float()
        box2_mask = (I_2 == 1).float()
        
        # =========== #
        #    Part 1   #
        # =========== #
        error_1_1 = self.mse(
            I_1 * (box1_mask * pred[..., 21:22] + box2_mask * pred[..., 26:27]),
            I_1 * true[..., 21:22]
        )
        error_1_2 = self.mse(
            I_1 * (box1_mask * pred[..., 22:23] + box2_mask * pred[..., 27:28]),
            I_1 * true[..., 22:23]
        )
        error_1_3 = self.mse(
            I_1 * (box1_mask * torch.sqrt(pred[..., 23:24]) + box2_mask * torch.sqrt(pred[..., 28:29])),
            I_1 * torch.sqrt(true[..., 23:24])
        )
        error_1_4 = self.mse(
            I_1 * (box1_mask * torch.sqrt(pred[..., 24:25]) + box2_mask * torch.sqrt(pred[..., 29:30])),
            I_1 * torch.sqrt(true[..., 24:25])
        )
        
        loss_1 = self.lambda_coord * (error_1_1 + error_1_2 + error_1_3 + error_1_4)
        
        # =========== #
        #    Part 2   #
        # =========== #
        loss_2 = self.mse(
            I_1 * (box1_mask * pred[..., 20:21] + box2_mask * pred[..., 25:26]),
            I_1 * true[..., 20:21]
        )
        
        # =========== #
        #    Part 3   #
        # =========== #
        error_3_1 = self.mse(
            (1 - I_1) * pred[..., 20:21],
            (1 - I_1) * true[..., 20:21]
        )
        error_3_2 = self.mse(
            (1 - I_1) * pred[..., 25:26],
            (1 - I_1) * true[..., 20:21]
        )
        loss_3 = self.lambda_noobj * (error_3_1 + error_3_2)
        
        # =========== #
        #    Part 4   #
        # =========== #
        loss_4 = self.mse(
            I_1 * pred[..., :20],
            I_1 * true[..., :20]
        )
        
        loss = loss_1 + loss_2 + loss_3 + loss_4
        
        return loss
    
    
def test():
    x, y, w, h = 0.4870000000000001, 0.9939939939939944, 5.026, 5.88588588588588
    
    pred = torch.zeros((1, 7, 7, 30))  # (batch_size, S, S, 5*B+C)
    pred[0, 3, 3, 13] = 1.  # class_label is 14
    pred[0, 3, 3, 20] = 1.  # confidence 
    pred[0, 3, 3, 21:25] = torch.tensor([x, y, w, h])  # box
    
    true = torch.zeros((1, 7, 7, 25))  # (batch_size, S, S, 5+C)
    true[0, 3, 3, 13] = 1.  # class_label is 14
    true[0, 3, 3, 20] = 1.  # confidence 
    true[0, 3, 3, 21:25] = torch.tensor([x, y, w, h])  # box
    
    loss = YOLOv1Loss()
    assert loss(pred, true) == 0., "Unexpected value."
    pred[0, 3, 3, 20] = 0.7987  # confidence 
    assert loss(pred, true) > 0., "Unexpected value."
    
    
if __name__ == "__main__":
    test()
    