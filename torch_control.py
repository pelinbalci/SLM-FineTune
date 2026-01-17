import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA used by torch:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))


import transformers, datasets, accelerate, trl, peft, huggingface_hub
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("huggingface_hub", huggingface_hub.__version__)
