"""
tf_transformers_imdb.py
Full end-to-end example: fine-tune a Hugging Face Transformer with TensorFlow (Keras)
on the IMDB sentiment dataset.

Usage:
    python tf_transformers_imdb.py
"""

import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import evaluate
import argparse

# -----------------------
# Config / Hyperparams
# -----------------------
MODEL_NAME = "distilbert-base-uncased"   # TF-compatible checkpoint
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./tf_distilbert_imdb"
USE_SUBSET = False           # set True for faster local tests
TRAIN_SUBSET = 20000         # if USE_SUBSET True, number of train examples
EVAL_SUBSET = 5000           # if USE_SUBSET True, number of eval examples
SEED = 42

# -----------------------
# Helpers
# -----------------------
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

# -----------------------
# Load dataset
# -----------------------
print("Loading dataset...")
raw_datasets = load_dataset("imdb")
if USE_SUBSET:
    raw_train = raw_datasets["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSET))
    raw_test = raw_datasets["test"].shuffle(seed=SEED).select(range(EVAL_SUBSET))
else:
    raw_train = raw_datasets["train"]
    raw_test = raw_datasets["test"]

print(f"Train size: {len(raw_train)}, Test size: {len(raw_test)}")

# -----------------------
# Tokenizer & Preprocessing
# -----------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess_batch(examples):
    # tokenizes a batch and returns padding/truncated tensors
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

print("Tokenizing dataset (this may take a while)...")
train_tokenized = raw_train.map(preprocess_batch, batched=True, remove_columns=["text"])
test_tokenized = raw_test.map(preprocess_batch, batched=True, remove_columns=["text"])

# keep only necessary columns and set format to numpy for generator
train_tokenized = train_tokenized.remove_columns([c for c in train_tokenized.column_names if c not in ("input_ids","attention_mask","label")])
test_tokenized = test_tokenized.remove_columns([c for c in test_tokenized.column_names if c not in ("input_ids","attention_mask","label")])

# -----------------------
# Convert to tf.data.Dataset
# -----------------------
print("Building tf.data pipelines...")

def gen_from_dataset(dataset):
    # generator yields (features, label) where features is dict of arrays
    for ex in dataset:
        features = {
            "input_ids": np.array(ex["input_ids"], dtype=np.int32),
            "attention_mask": np.array(ex["attention_mask"], dtype=np.int32),
        }
        label = np.int64(ex["label"])
        yield features, label

output_signature = (
    {
        "input_ids": tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
        "attention_mask": tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
    },
    tf.TensorSpec(shape=(), dtype=tf.int64)
)

train_tf = tf.data.Dataset.from_generator(lambda: gen_from_dataset(train_tokenized), output_signature=output_signature)
train_tf = train_tf.shuffle(2048, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_tf = tf.data.Dataset.from_generator(lambda: gen_from_dataset(test_tokenized), output_signature=output_signature)
test_tf = test_tf.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------
# Build / Load TF model
# -----------------------
print("Loading TF model...")
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Optionally enable mixed precision for speed on modern GPUs
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# -----------------------
# Callbacks
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7)
]

# -----------------------
# Train
# -----------------------
print("Starting training...")
history = model.fit(
    train_tf,
    validation_data=test_tf,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -----------------------
# Evaluate
# -----------------------
print("Loading evaluation metric...")
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

print("Evaluating on test set (batched inference)...")
# run batched inference and accumulate predictions
y_true = []
y_pred = []

for batch in test_tf:
    features, labels = batch
    logits = model(features, training=False)[0]   # (batch, num_labels)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.tolist())

acc = acc_metric.compute(predictions=y_pred, references=y_true)["accuracy"]
f1 = f1_metric.compute(predictions=y_pred, references=y_true, average="binary")["f1"]
print(f"Test Accuracy: {acc:.4f}, F1 (binary): {f1:.4f}")

# -----------------------
# Save the model (SavedModel format)
# -----------------------
print("Saving model as SavedModel to:", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)   # saves HF-style weights + config (TF)

# -----------------------
# Quick inference util
# -----------------------
def predict_texts(texts, top_k=1):
    toks = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="tf")
    logits = model(toks)[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    return preds, probs

if __name__ == "__main__":
    samples = [
        "That movie was amazing — I loved it and would watch again!",
        "Terrible film. I want my two hours back."
    ]
    p, pr = predict_texts(samples)
    for text, pred, prob in zip(samples, p, pr):
        print("TEXT:", text)
        print("PRED:", int(pred), "PROBS:", prob)
        print("----")
