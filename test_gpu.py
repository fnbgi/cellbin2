import onnxruntime

use_list = onnxruntime.get_available_providers()
GPU_KEY = 'CUDAExecutionProvider'
CPU_KEY = 'CPUExecutionProvider'
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']


if GPU_KEY in use_list:
    print("The environment supports the use of **GPU**.")
elif CPU_KEY not in use_list:
    print("The environment does not support **GPU**,but support **CPU**")
else:
    print("The environment does not support **GPU and CPU**")