import argparse
import logging
from time import time
from tqdm import tqdm
import cv2

import torch
from torchmetrics import F1Score
from model import inceptionv4
from dataset import ImageDataset
from torchvision import transforms 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', dest='model_path', type=str, default='check_points/inception/model_state_dict_best.pt')
    parser.add_argument('--data_path', dest='data_path', type=str, default='dataset/casting_data/test')
    args = parser.parse_args()
    
    
    logger.info('model loading..')
    num_classes = 1
    usb_tensorrt = False
    model_path = args.model_path
    data_path = args.data_path
    labels = ["normal", "defect"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inceptionv4(num_classes = num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    logger.info('dataset loading..')
    tranform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(data_path, labels, tranform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info('loaded dataset : {}'.format(len(dataset)))

    correct = 0
    f1 = F1Score(num_classes=2, threshold=0.5)
    start_time = time()
    preds = []
    targets = []
    logger.info('inferencing images..')
    total = len(dataset)
    cnt = 0
    with torch.no_grad():
        for path, data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = output[0,0]
            output = 1 if output[0,0] >= 0.5 else 0
            target = int(target[0])
            preds.append(output)
            targets.append(target)

            cnt += 1
            elap = time() - start_time
            fps = cnt / elap

            logger.info('{}/{} - {}, Predicted : {}, Actual : {}, Correct : {}, loss : {}'.format(cnt, total, path[0], labels[output], labels[target], output == target, loss))

    cv2.destroyAllWindows()

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    # acc = (correct/len(dataset))
    f1_score = f1(preds, targets) 
    
    elap = time() - start_time
    fps = total / elap
    logger.info('f1-score : {:.4f}, fps : {:.4f}'.format(float(f1_score), fps))