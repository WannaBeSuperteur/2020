import os
from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(device_lib.list_local_devices())
