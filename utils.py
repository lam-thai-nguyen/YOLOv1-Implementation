"""
Last edited on: Jul 11, 2024
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
    x1 = torch.max(coord1[..., 0:1], coord2[..., 0:1])
    y1 = torch.max(coord1[..., 1:2], coord2[..., 1:2])
    x2 = torch.min(coord1[..., 2:3], coord2[..., 2:3])
    y2 = torch.min(coord1[..., 3:4], coord2[..., 3:4])
    
    intersection_area = torch.clamp(x2-x1, min=0) * torch.clamp(y2-y1, min=0)
    
    # Areas
    area1 = (coord1[..., 2:3] - coord1[..., 0:1]) * (coord1[..., 3:4] - coord1[..., 1:2])
    area2 = (coord2[..., 2:3] - coord2[..., 0:1]) * (coord2[..., 3:4] - coord2[..., 1:2])
    
    # Union area
    union_area = area1 + area2 - intersection_area + 1e-6
    
    # IoU
    IoU = intersection_area / union_area
    
    return IoU 


def _box2coord(box):
    """
    (x, y, w, h) to (x1, y1, x2, y2)

    Args:
        box (tensor) (batch_size, 7, 7, 4)
        
    Returns:
        coord (tensor) (batch_size, 7, 7, 4)
    """
    x, y, w, h = box[..., 0:1], box[..., 1:2], box[..., 2:3], box[..., 3:4]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    coord = torch.cat((x1, y1, x2, y2), dim=-1)
    return coord


def test_IoU():
    box1, box2 = torch.randn((2, 7, 7, 4)), torch.randn((2, 7, 7, 4))
    iou = IoU(box1, box2)
    assert iou.size() == (2, 7, 7, 1)
    
    
def NMS(boxes, confidence_threshold, IoU_threshold):
    """
    Non-max suppression

    Args:
        boxes (list): each entry is [class_label, confidence, x, y, w, h]
        confidence_threshold (float): cutoff confidence
        IoU_threshold (float): cutoff IoU
    Returns:
        remaining_boxes (list)
    """
    # Confidence threshold
    boxes = [box for box in boxes if box[1] > confidence_threshold]
    boxes = sorted(boxes, key=lambda x : x[1], reverse=True)
    remaining_boxes = []
    
    # IoU threshold
    while boxes:
        # Pick the box with the highest confidence
        highest_confidence_box = boxes.pop(0)
        
        # Discard remaining boxes with IoU(highest_confidence_box, examined_box) >= IoU_threshold
        for examined_box in boxes:
            # If the examined box has the same class label as the highest_confidence_box
            if examined_box[0] == highest_confidence_box[0]:
                iou = IoU(torch.tensor(highest_confidence_box[2:]), torch.tensor(examined_box[2:]))
                if iou >= IoU_threshold:
                    boxes.remove(examined_box)
    
        remaining_boxes.append(highest_confidence_box)
    
    return remaining_boxes
    
    
def test_NMS():
    boxes = [
        [2, 0.8, 0.55, 0.66, 0.3, 0.5],
        [2, 0.9, 0.6, 0.7, 0.3, 0.5],
        [2, 0.3, 0., 0., 0., 0.],
        [3, 0.87, 0.6, 0.85, 0.6, 0.5],
        [3, 0.66, 0.65, 0.75, 0.6, 0.5],
    ]
    
    remaining_boxes = NMS(boxes, 0.6, 0.5)
    assert len(remaining_boxes) == 2
    

def mAP():
    ...
    
    
if __name__ == "__main__":
    test_IoU()
    test_NMS()
    