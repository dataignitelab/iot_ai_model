#!/bin/bash
trtexec --onnx=check_points/inception/model.onnx --saveEngine=check_points/inception/model.engine --verbose --fp16