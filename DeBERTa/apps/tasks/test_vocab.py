import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("/home/ec2-user/deberta_pretraining/DeBERTa/apps/tasks/rna_char.model")#spm.model")  # Replace with your actual .spm file path

# Get vocabulary size
vocab_size = sp.get_piece_size()

# Print the vocabulary

for i in range(vocab_size):
    piece = sp.id_to_piece(i)
    print(f"{i}\t{piece}")

