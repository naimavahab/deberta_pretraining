import torch, gc

#del some_tensor_or_model_component
torch.cuda.empty_cache()
gc.collect()

