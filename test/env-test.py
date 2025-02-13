import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
