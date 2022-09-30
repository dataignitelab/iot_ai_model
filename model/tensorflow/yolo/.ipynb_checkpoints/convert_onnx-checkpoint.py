import tensorflow as tf

# from model import resnet50
import tf2onnx

# from dataset import create_batch_generator

import logging
from time import time 

from yolo import createModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    # ./trtexec --onnx=/home/workspace/iot_ai_model/check_points/resnet50/model.onnx --saveEngine=/home/workspace/iot_ai_model/check_points/resnet50/model.engine --verbose
    
    input_size = 416
    num_class = 10
    anchors = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
    strides = [16, 32]
    xyscale = [1.05, 1.05]

    model = createModel(num_class, input_size, strides, anchors, xyscale)
    # model.load_weights( './check_points/yolo/400_best')
    model.load_weights('./check_points/yolo/yoloy_model.h5')
    
    input_signature = [tf.TensorSpec([1, input_size, input_size, 3], tf.float32, name='x')]
    onnx_model, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature)
    
    temp_model_file = 'check_points/yolo/model.onnx'
    
    with open(temp_model_file, "wb") as f:
        f.write(onnx_model.SerializeToString())