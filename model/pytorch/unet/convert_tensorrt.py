import os
import subprocess

def run():
    subprocess.run(["trtexec", 
                    "--onnx=check_points/unet/model.onnx",
                    "--saveEngine=check_points/unet/model.engine",
                    "--verbose",
                    "--workspace=512",
                    "--best"])

if __name__ == '__main__':
    run()