from gliner import GLiNERConfig, GLiNER
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/home/ec2-user/deberta_pretraining/DeBERTa/apps/tasks/deberta_rna_tokeniser", use_fast=False, local_files_only=True)

print(f"Tokenizer vocab size: {len(tokenizer)}")
#print(f"Model vocab size: {model.config.vocab_size}")

# Print all tokens in the tokenizer
for token, idx in tokenizer.get_vocab().items():
    print(f"{idx}: {token}")



model = GLiNER.from_pretrained("/home/ec2-user/deberta_pretraining/experiments/language_model/deberta-base", local_files_only=True) #urchade/gliner_sma
print(f"Model vocab size: {model.config.vocab_size}")


transformer = model.model.token_rep_layer.bert_layer.model



embedding_layer = model.model.token_rep_layer.bert_layer.model.embeddings.word_embeddings
#print("Embedding matrix shape:", embedding_layer.weight.shape)

# Print first 20 tokens and their IDs
vocab_size = embedding_layer.weight.shape[0]
for idx in range(min(vocab_size, 20)):
    token = tokenizer.convert_ids_to_tokens(idx)
    print(f"{idx}: {token}")

transformer.resize_token_embeddings(len(tokenizer))

embedding_layer = model.model.token_rep_layer.bert_layer.model.embeddings.word_embeddings

vocab_size = embedding_layer.weight.shape[0]
for idx in range(min(vocab_size, 20)):
    token = tokenizer.convert_ids_to_tokens(idx)
    print(f"{idx}: {token}")
