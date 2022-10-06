import tensorflow as tf

from model import resnet_50
import tf2onnx

# from dataset import create_batch_generator

import logging
from time import time 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    # ./trtexec --onnx=/home/workspace/iot_ai_model/check_points/resnet50/model.onnx --saveEngine=/home/workspace/iot_ai_model/check_points/resnet50/model.engine --verbose
    
    # labels = ['defect', 'normal']
    
    model = resnet_50(num_classes=1)
    model.load_weights('check_points/resnet50/model_lite.h5')
    
    input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name='x')]
    onnx_model, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature)
    
    temp_model_file = 'check_points/resnet50/model.onnx'
    
    with open(temp_model_file, "wb") as f:
        f.write(onnx_model.SerializeToString())