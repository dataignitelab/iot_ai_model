#!/bin/sh
cd /root/UNet/
pwd
export PYTHONPATH=/root/UNet:$PYTHONPATH
/root/caffe/build/tools/caffe train --solver=solver.prototxt
