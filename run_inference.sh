#!/bin/bash

cd /home/workspace/iot_ai_model/

v_display="false"
v_label_model="please choose one in [inception, yolo, ssd, rnn, lstm, unet, resnet, augmentation]"

if [ -z "$1" ]; then
    echo $v_label_model
fi

if [ "$2" = "--display" ]; then
    v_display="true"
fi

if [ "$1" = "inception" ]; then
    python3 ./model/pytorch/inception/run_tensorrt.py --display ${v_display}
elif [ "$1" = "yolo" ]; then
    python3 ./model/tensorflow/yolo/run_tensorrt.py --display ${v_display}
elif [ "$1" = "ssd" ]; then
    python3 ./model/tensorflow/ssd/run_tensorrt.py --display ${v_display}
elif [ "$1" = "rnn" ]; then
    python3 ./model/pytorch/rnn/predict.py 
elif [ "$1" = "lstm" ]; then
    python3 ./model/pytorch/lstm/predict.py 
elif [ "$1" = "unet" ]; then
    python3 ./model/pytorch/unet/run_tensorrt.py --display ${v_display}
elif [ "$1" = "resnet" ]; then
    python3 ./model/tensorflow/resnet/run_tensorrt.py --display ${v_display}
elif [ "$1" = "augmentation" ]; then
    v_num_option=""
    if [ -z "$2" ]; then
        v_num_option="--num=5"
    elif [ "$2" = "--num="* ]; then
        v_num_option=$2
    else
        echo "[err] unknown option. please input --num=[NUMER]"
        exit 0
    fi
    python3 ./model/tensorflow/ssd/test_augmentation.py ${v_num_option}
else
    echo "[err] unknown process name."
    echo $v_label_model
fi

exit 0
