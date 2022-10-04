import argparse
import logging
from time import time
from tqdm import tqdm
import cv2

import torch
from model import Unet
import torch.onnx

def convert_onnx(torch_model, output):
    x = torch.randn(1, 3, 256, 256, requires_grad=False).cuda()
    torch_out = torch_model(x)

    # 모델 변환
    torch.onnx.export(torch_model,               # 실행될 모델
                      x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      output,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                      input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                      output_names = ['output'], # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                    'output' : {0 : 'batch_size'}})
    
    
if __name__ == '__main__':
    # ./trtexec --onnx=/home/workspace/iot_ai_model/check_points/inception/inceptionv4.onnx --saveEngine=/home/workspace/iot_ai_model/check_points/inception/inceptionv4_trt.engine --verbose
    
    checkpoints_path = 'check_points/unet'
    model_path = f"{checkpoints_path}/model_state_dict_1000.pt"
    output = f'{checkpoints_path}/model.onnx'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet().to(device)
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    
    convert_onnx(model, output)