# IoT AI model project
  - IIoT 데이터를 활용한 AI 모델 개발 프로젝트 입니다.
  - Target Device : Nivida jetson TX2 (4GB)
  - 학습머신에서는 Pytorch, Tensorflow로 모델 개발 및 학습하고 ONNX로 전환하며, 최종적으로 Jetson TX2에서 ONNX를 TensorRT 모델로 전환하여 사용합니다.
  - 공개된 학습데이터셋을 활용하며, 본 프로젝트 코드에는 데이터셋은 포함되어 있지 않습니다. (아래 링크를 따라 별도로 다운로드)
 
## 이미지 분류 모델

#### 1. Pytorch 기반 Inception v4
  - 목적 : 주조 제품의 불량 검출 (Classification, 클래스 2종)
  - 학습 데이터 : 224x224 이미지, 5만장
    - https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
  - 목표 성능치 : 8.8 fps
  
#### 2. Tensorflow 기반 ResNet50
  - 목적 : 주조 제품의 불량 검출 (Classification, 클래스 2종)
  - 학습 데이터 : 224x224 이미지, 5만장
    - https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
  - 목표 성능치 : 28.8 fps

## 객체 인식 모델

#### 3. Tensorflow 기반 SSD ResNet18
  - 목적 : 디지털 배전판, 계기판 숫자 인식 (Text recongnition)
  - 학습 데이터 : 서버실 디지털 온도계, 습도계, 시계
    - https://www.kaggle.com/datasets/keshavaprasad/svhnvocyolodigitdetector?resource=download
  - 목표 성능치 : 12.8 fps
  
#### 4. Keras 기반 Yolo3
  - 목적 : 디지털 배전판, 계기판 숫자 인식 (Text recongnition)
  - 학습 데이터 : 서버실 디지털 온도계, 습도계, 시계
    - https://www.kaggle.com/datasets/keshavaprasad/svhnvocyolodigitdetector?resource=download  
  - 목표 성능치 : 20 fps

## 이미지 분할 모델 (Image Segmentation)

#### 5. Unet
  - 목적 : 이미지 내의 사람 영역 인식
  - 학습 데이터 : 다양한 크기와 자세인 사람(전신, 상반신) 이미지 (supervisely human)
    - https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
  - 목표 성능치 : 13.5 fps

## 시계열 분류 모델

#### 6. Pytorch 기반 LSTM
  - 목적 : 3상 모터의 전류 데이터 기반 이상 감지 (Classification, 클래스 5종)
  - 학습 데이터 : 지하철 환풍구 3상모터의 전류 데이터 
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=238
  - 목표 성능치 : F1-Score 90% 이상
  
#### 7. Pytorch 기반 RNN
  - 목적 : 3상 모터의 진동 데이터 기반 이상 감지 (Classification, 클래스 5종)
  - 학습 데이터 : 지하철 환풍구 3상모터의 진동 데이터 
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=238
  - 목표 성능치 : F1-Score 90% 이상
