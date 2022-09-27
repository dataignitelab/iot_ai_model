import tensorflow as tf

from model import resnet50
from dataset import create_batch_generator

import logging
from time import time 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    labels = ['defect', 'normal']
    
    model = tf.keras.models.load_model('check_points/resnet50/model.h5')
    
    
    gen, total = create_batch_generator('dataset/casting_data/test')
    
    start_time = time()
    elap = pre_elap = fps = .0
    preds = []
    targets = []
    cnt = 0
    for path, img, target in gen:
        output = model(img)
        
        # loss = output[0][0]
        output = 1 if output[0][0] >= 0.5 else 0
        target = int(target[0])
        
        preds.append(output)
        targets.append(target)

        cnt += 1
        
        logger.info('{}/{} - {}, Predicted : {}, Actual : {}, Correct : {}, fps: {:.1f}'.format(cnt, total, path[0], labels[output], labels[target], output == target, fps))
        
        elap = time() - start_time
        fps = max(0.0, 1.0 / (elap - pre_elap))
        pre_elap = elap
    
    # model.predict(
    #     normalized_ds,
    #     batch_size=1,
    #     verbose=1,
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False
    # )