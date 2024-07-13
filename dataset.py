"""
Last edited on: Jul 8, 2024
by: Lam Thai Nguyen
"""

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image and label
        img_name = os.path.join(self.root_dir, "images", self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        
        boxes_name = os.path.join(self.root_dir, "labels", self.annotations.iloc[idx, 1])
        boxes = []
        with open(boxes_name) as f:
            for line in f.readlines():
                boxes.append(line[:-2])
                
        if self.transform:
            image = self.transform(image)
            
        label = self._encode_box(boxes)
        
        return image, label
        
    
    def _encode_box(self, boxes):
        """
        Note: 
            Each value of a single bounding box is relative to the image height and width.
            As a result, each value has been normalized to the range of [0, 1], inclusively.
        Args:
            boxes (list[str]): box comes in the form of ['class_label x y w h', '...', ...], each str represents a bounding box
        Returns:
            label (tensor) ()
        """
        S = 7  # number of grid cells
        B = 2  # number of predictions per cell
        C = 20  # number of classes
        label = torch.zeros((S, S, C + 5))
        
        for box in boxes:
            values = box.split(" ")
            class_label = int(values[0])
            x, y, w, h = float(values[1]), float(values[2]), float(values[3]), float(values[4])
            
            # ===================================================================================== #
            #   Some math here:                                                                     #
            #       if   x/width=constant   and the width is divided into   s   equal segments,     #
            #       then   x/(width/s)=constant*s   meaning   x=constant*s * 1_cell_length          #
            #       so,   x_cell=x-int(constant*s)=constant*s-int(constant*s)                       #
            # ===================================================================================== #
            grid_x, grid_y = int(x * S), int(y * S)
            x_cell = x * S - grid_x
            y_cell = y * S - grid_y
            w_cell = w * S
            h_cell = h * S
            
            # Complete `label`
            label[grid_y, grid_x, class_label] = 1  # specify which class in in this cell
            label[grid_y, grid_x, 20] = 1  # specify the confidence
            label[grid_y, grid_x, 21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])  # box coordinates
            
        return label
        
        
def test():
    dataset = VOCDataset("data/8examples.csv", "data")
    assert len(dataset) == 8, "Unexpected length."
    img1, label1 = dataset[0]
    assert label1.size() == (7, 7, 25), "Unexpected size"
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img1)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
        
        
if __name__ == "__main__":
    test()
    