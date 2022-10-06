#!/bin/bash
trtexec --onnx=check_points/resnet50/model.onnx --saveEngine=check_points/resnet50/model.engine --verbose --best