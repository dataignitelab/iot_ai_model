import os
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import tensorflow as tf

# from box_utils import compute_iou


palette = [
    (255, 56, 56),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (255, 55, 199)
]


class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=1, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')
    
    def display_image(self, img, boxes, labels, name = 'img', wait=1):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            # top_left = (int(box[0]), int(box[1]))
            # bot_right = (int(box[2]), int(box[3]))

            color = palette[idx]

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.rectangle(image, (box[0], box[1]-10), (box[0]+10,box[1]), color, -1)
            cv2.putText(image, '{}'.format(cls_name), (box[0], box[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imshow(name, image)
        if wait > -1:
            cv2.waitKey(wait)
            
    def plt_image(self, img, boxes, labels, name = 'img', wait=1):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            # top_left = (int(box[0]), int(box[1]))
            # bot_right = (int(box[2]), int(box[3]))

            color = palette[idx]

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.rectangle(image, (box[0], box[1]-10), (box[0]+10,box[1]), color, -1)
            cv2.putText(image, '{}'.format(cls_name), (box[0], box[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

def padding(img, boxes = None, constant_values = 144, pad_type = 'constant'):
    img = np.array(img)
    h, w, _ = img.shape
    gap = abs(w - h) // 2
    w_gap = 0
    h_gap = 0
    if w > h:
        reshape_img = np.pad(img, ((gap, gap), (0, 0), (0,0)), pad_type, constant_values=constant_values)
        h_gap = gap
        new_h = h + (gap*2)
        new_w = w
    else:
        reshape_img = np.pad(img, ((0, 0), (gap, gap), (0,0)), pad_type, constant_values=constant_values)
        w_gap = gap
        new_w = w + (gap*2)
        new_h = h

    if boxes is not None:
        boxes = boxes * [w, h, w, h]    
        boxes = boxes + [w_gap, h_gap, w_gap, h_gap]
        boxes = boxes / [new_w, new_h, new_w, new_h]
        
    img = Image.fromarray(reshape_img)
    return img, boxes
            
# def generate_patch(boxes, threshold):
#     """ Function to generate a random patch within the image
#         If the patch overlaps any gt boxes at above the threshold,
#         then the patch is picked, otherwise generate another patch

#     Args:
#         boxes: box tensor (num_boxes, 4)
#         threshold: iou threshold to decide whether to choose the patch

#     Returns:
#         patch: the picked patch
#         ious: an array to store IOUs of the patch and all gt boxes
#     """
#     while True:
#         patch_w = random.uniform(0.1, 1)
#         scale = random.uniform(0.5, 2)
#         patch_h = patch_w * scale
#         patch_xmin = random.uniform(0, 1 - patch_w)
#         patch_ymin = random.uniform(0, 1 - patch_h)
#         patch_xmax = patch_xmin + patch_w
#         patch_ymax = patch_ymin + patch_h
#         patch = np.array(
#             [[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
#             dtype=np.float32)
#         patch = np.clip(patch, 0.0, 1.0)
#         ious = compute_iou(tf.constant(patch), boxes)
#         if tf.math.reduce_any(ious >= threshold):
#             break

#     return patch[0], ious[0]

def random_resize(img, boxes):
    w, h = img.size
    xy1_gap = min(min(boxes[:, 0]), min(boxes[:, 1]))
    xy2_gap = min(min(w - boxes[:, 2]), min(h - boxes[:, 3]))
    gap = min(xy1_gap, xy2_gap)

    # zoom in
    if random.random() < 0.5:
        ratio = random.uniform(0.1, 0.9)
        gap = gap * ratio

        pixel_x = int(w * gap)
        pixel_y = int(h * gap)
        
        img = img.crop((pixel_x, pixel_y, w-pixel_y, h-pixel_y))
        pixel_x = - pixel_x
        pixel_y = - pixel_y
    else: # zoom out
        ratio = random.uniform(0.1, 0.5)
        gap = gap * ratio

        pixel_x = int(w * gap)
        pixel_y = int(h * gap)
        
        np_img = np.array(img)
        np_img = np.pad(np_img, ((pixel_y, pixel_y), (pixel_x,pixel_x), (0,0)), mode='reflect')
        img = Image.fromarray(np_img)
        
    new_w, new_h = img.size
    boxes = (boxes * [w, h, w, h] + [pixel_x, pixel_y, pixel_x, pixel_y]) / [new_w, new_h, new_w, new_h]    
    return img, boxes

def random_brightness(img, factor = (0.5, 1.7)):
    enhancer = ImageEnhance.Brightness(img)

    if np.random.random() > 0.5:
        factor = np.random.uniform(factor[0], 0.8)
    
    else:
        factor = np.random.uniform(factor[0], 1.2)

    img = enhancer.enhance(factor)

    return img


def random_translate(img, boxes):
    w, h = img.size
    boxes = boxes * [w, h, w, h]
    boxes = boxes.astype(int)
    
    # 왼쪽의 여유공간을 오른쪽으로 할당. 나머지 공간도 반대쪽으로 할당
    right = min(boxes[:, 0])
    left = w - max(boxes[:, 2])
    bottom = min(boxes[:, 1])
    top = h - max(boxes[:, 3])
    
    pixel_x = left + right
    pixel_y = top + bottom
    
    result = Image.new(mode = img.mode, size = (w + pixel_x, h + pixel_y), color = (144,144,144))
    result.paste(img, (left, top))
    img = result
    
    new_x = int(pixel_x * random.uniform(0.1, 0.9))
    new_y = int(pixel_y * random.uniform(0.1, 0.9))
    img = result.crop((new_x, new_y, w + new_x, h + new_y))
    
    boxes = (boxes + [left - new_x, top - new_y, left - new_x, top - new_y]) / [w, h, w, h]
    return img, boxes


def random_patching(img, boxes, labels):
    """ Function to apply random patching
        Firstly, a patch is randomly picked
        Then only gt boxes of which IOU with the patch is above a threshold
        and has center point lies within the patch will be selected

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the cropped PIL Image
        boxes: selected gt boxes tensor (new_num_boxes, 4)
        labels: selected gt labels tensor (new_num_boxes,)
    """
    threshold = np.random.choice(np.linspace(0.1, 0.7, 4))

    patch, ious = generate_patch(boxes, threshold)

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep_idx = (
        (ious > 0.3) &
        (box_centers[:, 0] > patch[0]) &
        (box_centers[:, 1] > patch[1]) &
        (box_centers[:, 0] < patch[2]) &
        (box_centers[:, 1] < patch[3])
    )

    if not tf.math.reduce_any(keep_idx):
        return img, boxes, labels

    img = img.crop(patch)

    boxes = boxes[keep_idx]
    patch_w = patch[2] - patch[0]
    patch_h = patch[3] - patch[1]
    boxes = tf.stack([
        (boxes[:, 0] - patch[0]) / patch_w,
        (boxes[:, 1] - patch[1]) / patch_h,
        (boxes[:, 2] - patch[0]) / patch_w,
        (boxes[:, 3] - patch[1]) / patch_h], axis=1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    labels = labels[keep_idx]

    return img, boxes, labels


def horizontal_flip(img, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly

    Args:
        img: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        img: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels
