# ================================================================================= #
#   MODIFICATION: I use BatchNorm although the original paper didn't use BatchNorm  #
# ================================================================================= #

"""
Last edited on: Jul 6, 2024
by: Lam Thai Nguyen
"""

import torch
import torch.nn as nn


def Conv1x1(in_channels, out_channels, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.1)
    )
    
    
def Conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.1)
    )
    
    
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv3x3(in_channels=64, out_channels=192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv1x1(in_channels=192, out_channels=128),
            Conv3x3(in_channels=128, out_channels=256),
            Conv1x1(in_channels=256, out_channels=256),
            Conv3x3(in_channels=256, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv1x1(in_channels=512, out_channels=256),
            Conv3x3(in_channels=256, out_channels=512),
            Conv1x1(in_channels=512, out_channels=256),
            Conv3x3(in_channels=256, out_channels=512),
            Conv1x1(in_channels=512, out_channels=256),
            Conv3x3(in_channels=256, out_channels=512),
            Conv1x1(in_channels=512, out_channels=256),
            Conv3x3(in_channels=256, out_channels=512),
            Conv1x1(in_channels=512, out_channels=512),
            Conv3x3(in_channels=512, out_channels=1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv1x1(in_channels=1024, out_channels=512),
            Conv3x3(in_channels=512, out_channels=1024),
            Conv1x1(in_channels=1024, out_channels=512),
            Conv3x3(in_channels=512, out_channels=1024),
            Conv3x3(in_channels=1024, out_channels=1024),
            Conv3x3(in_channels=1024, out_channels=1024, stride=2),
            Conv3x3(in_channels=1024, out_channels=1024),
            Conv3x3(in_channels=1024, out_channels=1024),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten from (N, C, H, W) to (N, -1)
            nn.Linear(in_features=1024*7*7, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=7*7*30)  # the predictions are encoded as (SxSx(B*5+C)). Here, B=2, C=20.
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def test():
    X = torch.rand((1, 3, 448, 448))  # (N, C, H, W)
    yolov1 = YOLOv1()
    output = yolov1(X)
    assert output.size() == (1, 7*7*30), "unexpected output size"
   
    
if __name__ == "__main__":
    test()
