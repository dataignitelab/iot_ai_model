#!/bin/bash
trtexec --onnx=check_points/ssd/model.onnx --saveEngine=check_points/ssd/model.engine --verbose --fp16