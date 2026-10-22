
import torch
import torchvision
import pandas as pd
from PIL import Image

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Pandas version:", pd.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA version:", torch.version.cuda)

    x = torch.rand(3, 3, device="cuda")
    print("GPU tensor:")
    print(x)
else:
    print("Running on CPU")
