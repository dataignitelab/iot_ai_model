#!/bin/bash

trtexec --onnx=/home/workspace/iot_ai_model/check_points/resnet50/model.onnx --saveEngine=/home/workspace/iot_ai_model/check_points/resnet50/model.engine --verbose --best
