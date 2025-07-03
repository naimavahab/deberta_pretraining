import yaml
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import load_dataset,load_from_disk
import numpy as np
import torch
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# ====== Load hyperparameters from local config ======
with open("config.yaml", "r") as f:
    local_config = yaml.safe_load(f)

# Initialize wandb with loaded config
wandb.init(project="deberta-mlm-rna", config=local_config)
config = wandb.config

# ====== Logging setup ======
def timestamp():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger("MLM")

# ====== Load model and tokenizer ======
model_name = config.model_name
tokenizer = AutoTokenizer.from_pretrained("deberta_rna_tokeniser")
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# ====== Load dataset ======
dataset = load_from_disk("../rna_dataset_split")
train_dataset = dataset["train"] #.select(range(100))
eval_dataset = dataset["test"]  #.select(range(100))
#dataset = load_dataset('text', data_files='../rna_cleaned.txt')['train']
#dataset = dataset.train_test_split(test_size=0.2, seed=42)
#train_dataset = dataset['train']
#eval_dataset = dataset['test']

# ====== Tokenization ======
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ====== Data collator ======
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=config.mlm_probability,
)

# ====== Metrics ======
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    true_labels = labels[mask].flatten()
    pred_labels = predictions[mask].flatten()

    accuracy = (true_labels == pred_labels).mean()
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits[mask]), torch.tensor(labels[mask]), reduction='mean'
    ).item()
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    perplexity = np.exp(loss)

    logger.info(f"{timestamp()}| accuracy = {accuracy}")
    logger.info(f"{timestamp()}| eval_loss = {loss}")
    logger.info(f"{timestamp()}| perplexity = {perplexity}")
    logger.info(f"{timestamp()}| precision = {precision}")
    logger.info(f"{timestamp()}| recall = {recall}")
    logger.info(f"{timestamp()}| f1 = {f1}")
    logger.info(f"{timestamp()}| mcc = {mcc}")

    return {
        "accuracy": accuracy,
        "eval_loss": loss,
        "perplexity": perplexity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }

# ====== Training arguments ======
training_args = TrainingArguments(
    output_dir="./deberta-mlm-sweep-output",
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.per_device_train_batch_size,
    num_train_epochs=config.num_train_epochs,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to=["wandb"],
)

# ====== Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ====== Train and evaluate ======
#trainer.train()
trainer.evaluate()

# ====== Save final model ======
output_dir = "./deberta_mlm_finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

