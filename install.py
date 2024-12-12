import subprocess
import sys
cuda_v = None
if len(sys.argv) > 1:
    cuda_v = sys.argv[1]

# subprocess.call(["conda", "install", "--channel", "conda-forge", "pyvips"])

with open('requirements.txt') as f:
    lines = f.readlines()
    for line in lines:
        package_name = line.strip()
        if not package_name.startswith("#"):
            if "onnxruntime" in package_name and cuda_v is not None:
                if int(cuda_v) == 12:
                    package_name = package_name.replace("1.15.1", "1.19.0")
            print(f"Installing {package_name}")
            # subprocess.call(["pip", "install", package_name])

