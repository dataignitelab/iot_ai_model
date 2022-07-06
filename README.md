# didimdol_ai_model

  - IIoT 데이터를 활용한 AI 모델 코드 입니다.
  - Target Device : Nivida jetson TX2

## 모델 종류

### 이미지 분류
1. Pytorch 기반 Inception v4
  - 목적 : 주조 제품의 불량 검출 (클래스 2종)
  - 학습 데이터 : 224x224 이미지, 5만장
  - 목표 성능치 : 8.8 fps
2. Tensorflow 기반 ResNet50
  - 목적 : 주조 제품의 불량 검출 (클래스 2종)
  - 학습 데이터 : 224x224 이미지, 5만장
  - 목표 성능치 : 28.8 fps

### 객체 인식
3. Tensorflow 기반 SSD ResNet18
  - 목적 : 디지털 배전판, 계기판 숫자 인식 (Object Detection, OCR)
  - 학습 데이터 : 서버실 디지털 온도계, 습도계, 시계
  - 목표 성능치 : 12.8 fps
4. Keras 기반 Yolo3
  - 목적 : 디지털 배전판, 계기판 숫자 인식 (Object Detection, OCR)
  - 학습 데이터 : 서버실 디지털 온도계, 습도계, 시계
  - 목표 성능치 : 20 fps
5. Caffe 기반 Unet
  - 목적 : 디지털 배전판, 계기판 숫자 인식 (Object Detection, OCR)
  - 학습 데이터 : 서버실 디지털 온도계, 습도계, 시계
  - 목표 성능치 : 13.5 fps

### 시계열 분류
6. Pytorch 기반 LSTM
  - 목적 : 3상 모터의 전류 데이터 기반 이상 감지 (클래스 5종)
  - 학습 데이터 : 지하철 환풍구 3상모터의 전류 데이터 
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=238
  - 목표 성능치 : F1-Score 95% 이상
7. Pytorch 기반 RNN
  - 목적 : 3상 모터의 진동 데이터 기반 이상 감지 (클래스 5종)
  - 학습 데이터 : 지하철 환풍구 3상모터의 진동 데이터 
    - https://aihub.or.kr/aihubdata/data/view.do?currMenu=116&topMenu=100&aihubDataSe=ty&dataSetSn=238
  - 목표 성능치 : F1-Score 95% 이상
