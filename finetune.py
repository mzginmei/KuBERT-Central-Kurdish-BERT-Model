import pandas as pd
import torch
import numpy as np

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score


# =========================
# CONFIG
# =========================

MODEL_NAME = "asosoft/KuBERT-Central-Kurdish-BERT-Model"

TRAIN_PATH = "train.csv"
VAL_PATH = "validation.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD DATASETS
# =========================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


# =========================
# REMOVE EMPTY COMMENTS
# =========================

train_df = train_df.dropna(subset=["comment"])
val_df = val_df.dropna(subset=["comment"])

train_df["comment"] = train_df["comment"].astype(str)
val_df["comment"] = val_df["comment"].astype(str)


# =========================
# RENAME LABEL COLUMN
# HuggingFace expects: labels
# =========================

train_df = train_df.rename(columns={"vote": "labels"})
val_df = val_df.rename(columns={"vote": "labels"})


# =========================
# KEEP ONLY NEEDED COLUMNS
# =========================

train_df = train_df[["comment", "labels"]]
val_df = val_df[["comment", "labels"]]


# =========================
# CONVERT TO HF DATASET
# =========================

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

from transformers import logging
logging.set_verbosity_error()

# =========================
# LOAD TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)


# =========================
# TOKENIZATION
# =========================

def tokenize(batch):

    return tokenizer(
        batch["comment"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)


# =========================
# FORMAT FOR PYTORCH
# =========================

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)


# =========================
# LOAD MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
).to(DEVICE)


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(
    output_dir="./kubert_finetuned",

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    learning_rate=2e-5,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=50,

    load_best_model_at_end=True
)


# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)


# =========================
# START TRAINING
# =========================

trainer.train()


# =========================
# FINAL EVALUATION
# =========================

trainer.evaluate()


# =========================
# SAVE MODEL
# =========================

trainer.save_model("kubert_hate_finetuned")

tokenizer.save_pretrained("kubert_hate_finetuned")


print("Training completed.")