# import tensorflow as tf
import numpy as np

def tensor_scatter_nd_update(tensor, indices, updates):
    for idx, coord in enumerate(indices):
        tensor[tuple(coord)] = updates[idx]
    return tensor

def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = np.minimum(np.maximum(bot_right - top_left, 0.0), 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = np.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = np.expand_dims(boxes_b, 0)
    top_left = np.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = np.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap    

def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.40):
    """ Compute regression and classification targets
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        gt_boxes: tensor (num_gt, 4)
                  of format (xmin, ymin, xmax, ymax)
        gt_labels: tensor (num_gt,)
    Returns:
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    """
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    transformed_default_boxes = transform_center_to_corner(default_boxes)
    iou = compute_iou(transformed_default_boxes, gt_boxes)
    
    best_gt_iou = np.max(iou,1)
    best_gt_idx = np.argmax(iou, 1)

    best_default_iou = np.max(iou, 0)
    best_default_idx = np.argmax(iou, 0)

    best_gt_idx = tensor_scatter_nd_update(
        best_gt_idx,
        np.expand_dims(best_default_idx, axis=1),
        np.arange(best_default_idx.shape[0], dtype=np.int64))

    # Normal way: use a for loop
    # for gt_idx, default_idx in enumerate(best_default_idx):
    #     best_gt_idx = tf.tensor_scatter_nd_update(
    #         best_gt_idx,
    #         tf.expand_dims([default_idx], 1),
    #         [gt_idx])

    best_gt_iou = tensor_scatter_nd_update(
        best_gt_iou,
        np.expand_dims(best_default_idx, 1),
        np.ones_like(best_default_idx, dtype=np.float32))

    gt_confs = np.take(gt_labels, best_gt_idx, axis=0)
    
    # for l in gt_labels:
    #     print(l, max(best_gt_iou[gt_confs == l]))
    
    gt_confs = np.where(
        [best_gt_iou < iou_threshold],
        np.zeros_like(gt_confs),
        gt_confs)
    
    # for l in gt_labels:
    #     print(l, len(gt_confs[gt_confs == l]))
    gt_boxes = np.take(gt_boxes, best_gt_idx, axis=0)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs


def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = transform_corner_to_center(boxes)

    locs = np.concatenate([
        (transformed_boxes[..., :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] * variance[0]),
        np.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]
    ], axis=-1)

    return locs


def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = np.concatenate([
        locs[..., :2] * variance[0] * default_boxes[:, 2:] + default_boxes[:, :2],
        np.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]
    ], axis=-1)

    boxes = transform_center_to_corner(locs)

    return boxes


def transform_corner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2, 
        boxes[..., 2:] - boxes[..., :2]
    ], axis=-1)

    return center_box


def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = np.concatenate([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2
    ], axis=-1)

    return corner_box


def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    selected = [0]
    idx = (-scores).argsort()
    idx = idx[:limit]
    boxes = np.take(boxes, idx, axis=0)

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = np.where(
            np.expand_dims(np.logical_not(next_indices), 0),
            np.ones_like(iou, dtype=np.float32),
            iou)

        if not np.any(next_indices):
            break

        selected.append((-(next_indices.astype(np.int32))).argsort()[0])

    return np.take(idx, selected, axis=0)
