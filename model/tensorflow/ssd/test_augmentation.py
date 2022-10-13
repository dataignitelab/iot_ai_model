from image_utils import ImageVisualizer
from dataset import VOCDataset
import image_utils as util 
import numpy as np
import cv2
import argparse

def get_coord(image, boxes):
    h,w = image.size
    new_boxes = np.array(boxes * [w, h, w, h], dtype=np.int16)
    return new_boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='augmentation')
    parser.add_argument('--num', default=5, type=int)
    args = parser.parse_args()
    
    count = args.num
    
    print(f'Testing number of images : {count}')
    
    LABEL_NAME = ['0','1','2','3','4','5','6','7','8','9']

    visualizer = ImageVisualizer(LABEL_NAME, save_dir='check_points/yolo/outputs/images')
    dataset = VOCDataset('dataset/server_room/test_digit.txt', None, 512)

    for idx in range(len(dataset)):
        image_o = dataset._get_image(idx)
        boxes_o, labels_o = dataset._get_annotation(idx, image_o.size)

        boxes = get_coord(image_o, boxes_o)
        visualizer.display_image(image_o, boxes, labels_o, 'original_image', wait = -1)
        print(f'{idx} loaded original_image')
        print(boxes)

        image, boxes = util.random_resize(image_o, boxes_o)
        boxes = get_coord(image, boxes)
        visualizer.display_image(image, boxes, labels_o, 'random_zomm_in', wait = -1)
        print(f'{idx} loaded random_zomm_in')
        print(boxes)

        image, boxes = util.random_translate(image_o, boxes_o)
        boxes = get_coord(image, boxes)
        visualizer.display_image(image, boxes, labels_o, 'random_translate', wait = -1)
        print(f'{idx} loaded random_translate')
        print(boxes)


        image = util.random_brightness(image_o)
        boxes = get_coord(image, boxes_o)
        visualizer.display_image(image, boxes, labels_o, 'random_brightness', wait = -1)
        print(f'{idx} loaded random_brightness')
        print(boxes)

        image, boxes, labels = util.horizontal_flip(image_o, boxes_o, labels_o)
        boxes = get_coord(image, boxes)
        visualizer.display_image(image, boxes, labels, 'random_flip', wait = -1)
        print(f'{idx} loaded random_flip')
        print(boxes)

        cv2.waitKey(0)

        if idx > (count-1): break

    cv2.destroyAllWindows()  