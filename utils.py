"""
Last edited on: Jul 12, 2024
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
    

def mAP(pred_boxes, true_boxes, IoU_threshold=0.5, num_classes=20):
    """
    Mean average precision

    Args:
        pred_boxes (list): each entry is [image_id, class_label, confidence, x, y, w, h]
        true_boxes (list): each entry is [image_id, class_label, confidence, x, y, w, h]
        IoU_threshold (float, optional): Defaults to 0.5.
        num_classes (int, optional): Defaults to 20.
    Returns:
        mAP (float): mAP for all classes @ specified IoU threshold
    """
    average_precision = []  # Each entry is a float representing AP for a single class
    
    for c in range(num_classes):
        # Capture all boxes of class c
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        if len(ground_truths) == 0:
            continue
        
        # The amount of true boxes in each image of class c is stored on a dictionary
        num_true_boxes = {}
        for box in ground_truths:
            image_id = box[0]
            if image_id not in num_true_boxes:
                num_true_boxes[image_id] = 0
            num_true_boxes[image_id] += 1
        
        # Sort detections by confidence score
        detections = sorted(detections, key=lambda x : x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        
        # Initialize matched ground truths
        matched_gt = []
        
        # Assigning each box to TP or FP
        for idx, detection in enumerate(detections):
            image_id, _, confidence, x, y, w, h = detection
            
            # ================================================================== #
            # Each detection's IoU is the best_iou out of all IoUs               #
            # between the detected box and all ground truths boxes of that image #          
            # ================================================================== #
            best_iou = 0. 
            best_gt = None 
            
            for gt in ground_truths:
                if gt[0] == image_id and gt not in matched_gt:
                    iou = IoU(torch.tensor([x, y, w, h]), torch.tensor(gt[3:]))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                        
            if best_iou > IoU_threshold:
                TP[idx] = 1
                matched_gt.append(best_gt)
            else:
                FP[idx] = 1
                
        # Calculate precision and recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / len(ground_truths)
        
        # Calculate the average precision (AP) using trapezoidal rule
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        average_precision.append(torch.trapz(precisions, recalls))
            
    mAP = sum(average_precision) / len(average_precision)
    
    return mAP
   
   
def test_mAP():
    pred_boxes = [
        # image_id, class_label, confidence, x, y, w, h
        [0, 0, 0.9, 0.5, 0.5, 0.2, 0.2],  # High confidence box, should be TP
        [0, 0, 0.8, 0.5, 0.5, 0.2, 0.2],  # Second box, should be FP due to duplicate detection
        [0, 0, 0.3, 0.1, 0.1, 0.1, 0.1],  # Low confidence, should be FP
    ]

    true_boxes = [
        # image_id, class_label, confidence, x, y, w, h
        [0, 0, 1.0, 0.5, 0.5, 0.2, 0.2],  # True positive box
        [0, 0, 1.0, 0.1, 0.1, 0.1, 0.1],  # True positive box
    ]
    mAP_value = mAP(pred_boxes, true_boxes, IoU_threshold=0.5, num_classes=2)
    assert 0. <= mAP_value <= 1.
     
    
if __name__ == "__main__":
    test_IoU()
    test_NMS()
    test_mAP()
    