#!/bin/bash
trtexec --onnx=check_points/yolo/model.onnx --saveEngine=check_points/yolo/model.engine --verbose --fp16