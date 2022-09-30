import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np

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
            idx = labels[i]
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            
            color = palette[idx]
            color = (color[0] / 255., color[1] / 255., color[2] / 255.)
            
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2]-box[0], box[3]-box[1],
                linewidth=1, edgecolor=color,
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')
    
    def display_image(self, img, boxes, labels, name):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i, box in enumerate(boxes):
            idx = labels[i] 
            cls_name = self.idx_to_name[idx]
            # top_left = (int(box[0]), int(box[1]))
            # bot_right = (int(box[2]), int(box[3]))

            color = palette[idx]

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.rectangle(image, (box[0], box[1]-10), (box[0]+10,box[1]), color, -1)
            cv2.putText(image, '{}'.format(cls_name), (box[0], box[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.imshow('img', image)
            cv2.waitKey(1)

        