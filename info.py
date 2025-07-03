from transformers import BertModel  # or any other HF model
import torch

model = BertModel.from_pretrained('microsoft/deberta-v3-large')

def print_model_layers(model):
    total_params = 0
    print(f"{'Layer Name':60} {'# Parameters':>15}")
    print("="*80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:60} {param_count:15,}")
    
    print("="*80)
    print(f"{'Total Parameters':60} {total_params:15,}")

print_model_layers(model)

