import argparse
import logging
from time import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms 

from model import inceptionv4
from dataset import ImageDataset

import torch.onnx

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

checkpoints_path = './didimdol/checkpoints/inception'

def convert_onnx():
    

def fit(model, dataloader, criterion, optimizer, device, half = False):
    loss = .0
    acc = .0
    correct = 0
    start_time = time()
    
    progress = tqdm(dataloader)
    for data, target in progress:
        data = data.to(device)
        if half :
            data = data.half()
        target = target.to(device)
        output = model(data)
        output = output.squeeze(dim=1)
        loss = criterion(output, target)
        
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss += loss
        
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        correct += output.eq(target).int().sum()

    acc = (correct/len(dataloader.dataset))
    loss = loss/len(dataloader.dataset)
    logger.info("{}, duration:{:6.1f}s, acc:{:.4f}, loss:{:.4f}".format(('trn' if model.training else 'val'), 
                                                                         time()-start_time, 
                                                                         acc, 
                                                                         loss ))
    return float(loss), float(acc)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='train..')
    parser.add_argument('--dataset', dest='dataset', type=str, default='dataset/casting_data/train')

    args = parser.parse_args()

    # training args 
    lr = 0.01
    batch_size = 128
    worker = 3
    num_classes = 1
    epochs = 20
    early_stopping = 5
    labels = ["normal", "defect"]

    dataset_path = args.dataset

    # datset
    dataset = ImageDataset(dataset_path, labels)

    total_size = len(dataset)
    trn_size = int(total_size * 0.7)
    val_size = total_size - trn_size

    trn_dataset, val_dataset = random_split(dataset, [trn_size, val_size])

    trn_dataset.dataset.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset.dataset.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=worker)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=worker)

    logger.info(f'trn: {len(trn_dataset)}, val: {len(val_dataset)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Load on {device}')

    model = inceptionv4(num_classes = num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)

    trn_loss = []
    trn_acc = []
    val_loss = []
    val_acc = []

    half = False

    min_loss = 99999.
    early_count = 0
    for epoch in range(1, epochs+1):
        logger.info(f'epoch {epoch}')

        model.train()
        loss, acc = fit(model, trn_dataloader, criterion, optimizer, device, half=half)

        trn_loss.append(loss)
        trn_acc.append(acc)

        model.eval()
        with torch.no_grad():
            loss, acc = fit(model, val_dataloader, criterion, optimizer, device, half=half)

            if loss >= min_loss:
                early_count += 1
                if early_count >= early_stopping:
                    break
            else:
                min_loss = loss
                early_count = 0

            if len(val_acc) > 0 and max(val_acc) < acc:
                torch.save(model.state_dict(), f"{checkpoints_path}/model_state_dict_{epoch}_best.pt")

            val_loss.append(loss)
            val_acc.append(acc)
            
    torch.save(model.state_dict(), f"{checkpoints_path}/model_state_dict_{epoch}.pt")