from transformers import AutoModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = AutoModel.from_pretrained('')

print(model)

total_params = 0

for name, param in model.named_parameters():
    print(name, param.shape)
    total_params += param.numel()

print(f"Total number of parameters in the model: {total_params}")
