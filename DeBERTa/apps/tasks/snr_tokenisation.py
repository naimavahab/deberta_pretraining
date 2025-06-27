import sentencepiece as spm

# Create a corpus with only A, U, C, G
with open("rna.txt", "w") as f:
    f.write("AUCGAUCGGAUCGAUC")

# Train with character-level model
spm.SentencePieceTrainer.train(
    input='rna.txt',
    model_prefix='rna_char',
    model_type='unigram',
    vocab_size=6,  # A, U, C, G, UNK, ▁ (space marker)
    character_coverage=1.0,
    unk_piece='[UNK]',
            # Make sure unk_id is set
#    unk_piece='<unk>',
    bos_id=-1,  # No <s>
    eos_id=-1,  # No </s>
 #   unk_id=-1,  # No <unk>
    pad_id=-1,  # No <pad>
    hard_vocab_limit=False
)

from transformers import AutoTokenizer, DebertaV2Tokenizer

tokenizer_deberta = DebertaV2Tokenizer(
    vocab_file  = "rna_char.model",
    max_len = 512,
)
path = 'deberta_rna_tokeniser'
tokenizer_deberta.save_pretrained(path )
tokenizer = DebertaV2Tokenizer.from_pretrained(
    path
)
for i, (token, idx) in enumerate(tokenizer.get_vocab().items()):
    if i < 20:
        print(f"{idx}: {token}")


'''
# Load and test
sp = spm.SentencePieceProcessor()
sp.load("rna_char.model")

print(sp.encode("AUGCU", out_type=str))  # e.g., ['▁', 'A', 'U', 'G', 'C', 'U']
'''
