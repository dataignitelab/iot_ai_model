import itertools
import math
import tensorflow as tf
import numpy as np


def generate_default_boxes(config, use_tensor = True):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    scales = config['scales'] # [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
    fm_sizes = config['fm_sizes'] # [38, 19, 10, 5, 3, 1]
    ratios = config['ratios'] # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    for m, fm_size in enumerate(fm_sizes): 
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    if use_tensor:
        default_boxes = tf.constant(default_boxes)
        default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)
    else:
        default_boxes = np.minimum(np.maximum(default_boxes, 0.0), 1.0)

    return default_boxes

