import torch
from transformers import DebertaForMaskedLM, DebertaConfig

# Load the original DeBERTa model checkpoint (Microsoft format)
state_dict = torch.load("rna_char.model", map_location="cpu")
print(state_dict)
# Create a matching Hugging Face DeBERTa config
config = DebertaConfig(
    vocab_size=50265,        # Set based on your tokenizer
    hidden_size=768,         # Checkpoint-specific
    num_attention_heads=12,  # Match original model
    num_hidden_layers=12,    # Match original model
    intermediate_size=3072,  # Usually 4x hidden_size
    type_vocab_size=0,       # DeBERTa typically doesn't use token type embeddings
)

# Initialize HF DeBERTa model
model = DebertaForMaskedLM(config)

# Load weights into HF model
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

