import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should show your CUDA version
print(torch.backends.cudnn.enabled)  # Should return True
print(torch.cuda.device_count())  # Should show number of GPUs
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
