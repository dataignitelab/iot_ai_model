#!/bin/sh
cd /home/workspace/iot_ai_model

pwd

export PYTHONPATH=/home/workspace/iot_ai_model:$PYTHONPATH
/root/caffe/build/tools/caffe train --solver=solver.prototxt
