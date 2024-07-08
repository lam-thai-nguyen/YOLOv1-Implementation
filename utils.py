"""
Last edited on: Jul 8, 2024
by: Lam Thai Nguyen
"""

import torch


def IoU(box1, box2):
    """
    Returns the Intersection over Union score

    Args:
        box1 (tensor) (batch_size, 7, 7, 4): box1[0] = (x, y, w, h)
        box2 (tensor) (batch_size, 7, 7, 4): box2[0] = (x, y, w, h)
    
    Returns:
        IoU (tensor) (batch_size, 7, 7, 1)
    """
    coord1, coord2 = _box2coord(box1), _box2coord(box2)
    
    # Intersection box
    x1 = torch.max(coord1[..., 0], coord2[..., 0])
    y1 = torch.max(coord1[..., 1], coord2[..., 1])
    x2 = torch.min(coord1[..., 2], coord2[..., 2])
    y2 = torch.min(coord1[..., 3], coord2[..., 3])
    
    intersection_area = torch.clamp(x2-x1, min=0) * torch.clamp(y2-y1, min=0)
    
    # Areas
    area1 = (coord1[..., 2] - coord1[..., 0]) * (coord1[..., 3] - coord1[..., 1])
    area2 = (coord2[..., 2] - coord2[..., 0]) * (coord2[..., 3] - coord2[..., 1])
    
    # Union area
    union_area = area1 + area2 - intersection_area + 1e-6
    
    # IoU
    IoU = (intersection_area / union_area).unsqueeze(-1)
    
    return IoU 


def _box2coord(box):
    """
    (x, y, w, h) to (x1, y1, x2, y2)

    Args:
        box (tensor) (batch_size, 7, 7, 4)
        
    Returns:
        coord (tensor) (batch_size, 7, 7, 4)
    """
    x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    coord = torch.stack((x1, y1, x2, y2), dim=-1)
    return coord


def test_IoU():
    box1, box2 = torch.randn((2, 7, 7, 4)), torch.randn((2, 7, 7, 4))
    iou = IoU(box1, box2)
    assert iou.size() == (2, 7, 7, 1)
    
    
if __name__ == "__main__":
    test_IoU()
    