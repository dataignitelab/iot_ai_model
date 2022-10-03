from glob import glob
from tqdm import tqdm
from time import time
import argparse
import logging
import os

from model import Unet
from dataset import load_image
from tensorrt_model import TrtModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def dice_loss(inputs, targets, smooth=1):
    inputs[inputs > 0.5] = 1
    inputs[inputs <= 0.5] = 0
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  

    return 1 - dice 

def inference(model_path, data_path, display = False):
    logger.info('model loading.. {}'.format(model_path))
    batch_size = 1
     # os.path.join("..","models","main.trt")
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)
    
    # data_paths = glob(dataset_path)
    
    
    logger.info('dataset loading..')
   
    with open(data_path, 'r') as f:
        line = f.readlines()

    total = len(line)
    logger.info('number of test dataset : {}'.format(total))
    
    logger.info('start inferencing')
    preds = []
    targets = []
    cnt = 0
    
    
    base_dir = os.path.dirname(data_path)
    
    start_time = time()
    pre_elap = 0.0
    fps = 0.0
    cost = .0
    for idx, row in enumerate(line):
        img_path, mask_path = row.rstrip().split(',')
        
        img = load_image(os.path.join(base_dir, img_path))
        mask = load_image(os.path.join(base_dir, mask_path))
        
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        output = model(img)
        
        output = output[0].reshape(img.shape)
        
        # loss = output[0][0]
        # output = 1 if output[0][0] >= 0.5 else 0
        # target = int(target[0])
        # preds.append(output)
        # targets.append(target)
        
        loss = dice_loss(img, output)
        
        cost += loss
        
        logger.info('{}/{} - {},  fps: {:.1f}'.format(idx+1, total, img_path, fps))

        if(display):
            img = cv2.imread(path[0])

            cv2.putText(img, 'Result: {}, Correct: {} '.format(labels[output], output == target), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.putText(img, 'FPS: {:.2f}'.format(fps), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.imshow('img', img)
            cv2.waitKey(1)
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
        
    if(display):
        cv2.destroyAllWindows()

    # preds = torch.tensor(preds)
    # targets = torch.tensor(targets)
    # # acc = (correct/len(dataset))
    # f1_score = f1(preds, targets) 
    
    elap = time() - start_time
    fps = total / elap
    logger.info('dice efficient: {:.4f}, fps: {:.4f}'.format(cost/total, fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unet')
    
    parser.add_argument('--model-path', dest='model_path', type=str, default='check_points/unet/model.engine')
    parser.add_argument('--data-path', dest='data_path', type=str, default='dataset/supervisely_person/test_data_list.txt')
    parser.add_argument('--display', dest='display', type=str2bool, default=False)
    
    args = parser.parse_args()
    logger.info(args)
    inference(args.model_path, args.data_path, args.display)