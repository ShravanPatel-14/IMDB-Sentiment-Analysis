# # src/data.py
# import numpy as np
# import tensorflow as tf
# from datasets import load_dataset
# from transformers import AutoTokenizer

# def get_tokenizer(model_name="distilbert-base-uncased", model_dir=None, max_len=128):
#     if model_dir:
#         return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
#     return AutoTokenizer.from_pretrained(model_name, use_fast=True)

# def preprocess_batch(examples, tokenizer, max_len=128):
#     """Return tokenized fields but do NOT remove original text here."""
#     out = tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=max_len,
#     )
#     # Keep original text as-is in the dataset so we can access it later
#     out["text"] = examples["text"]
#     return out

# def dataset_to_tf(train_or_ds, max_len=128, batch_size=16, shuffle=False, seed=42):
#     def gen():
#         for ex in train_or_ds:
#             features = {
#                 "input_ids": np.array(ex["input_ids"], dtype=np.int32),
#                 "attention_mask": np.array(ex["attention_mask"], dtype=np.int32),
#             }
#             label = np.int64(ex["label"])
#             yield features, label

#     output_signature = (
#         {
#             "input_ids": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
#             "attention_mask": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
#         },
#         tf.TensorSpec(shape=(), dtype=tf.int64)
#     )

#     tf_ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
#     if shuffle:
#         tf_ds = tf_ds.shuffle(2048, seed=seed)
#     tf_ds = tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return tf_ds

# def dataset_to_tf_with_text(ds, max_len=128, batch_size=16, shuffle=False, seed=42):
#     def gen():
#         for ex in ds:
#             features = {
#                 "input_ids": np.array(ex["input_ids"], dtype=np.int32),
#                 "attention_mask": np.array(ex["attention_mask"], dtype=np.int32),
#                 "text": ex["text"],   # keep Python str
#             }
#             label = np.int64(ex["label"])
#             yield features, label

#     output_signature = (
#         {
#             "input_ids": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
#             "attention_mask": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
#             "text": tf.TensorSpec(shape=(), dtype=tf.string),
#         },
#         tf.TensorSpec(shape=(), dtype=tf.int64)
#     )

#     tf_ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
#     if shuffle:
#         tf_ds = tf_ds.shuffle(2048, seed=seed)
#     tf_ds = tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return tf_ds


# def load_imdb_tokenized(model_name="distilbert-base-uncased", model_dir=None,
#                         max_len=128, use_subset=True, train_subset=2000, eval_subset=1000, seed=42):
#     """
#     Returns: train_tokenized, test_tokenized, tokenizer
#     Each tokenized dataset contains input_ids, attention_mask, label, text
#     """
#     raw = load_dataset("imdb")
#     if use_subset:
#         train = raw["train"].shuffle(seed=seed).select(range(train_subset))
#         test = raw["test"].shuffle(seed=seed).select(range(eval_subset))
#     else:
#         train = raw["train"]
#         test = raw["test"]

#     tokenizer = get_tokenizer(model_name=model_name, model_dir=model_dir, max_len=max_len)
#     train_t = train.map(lambda ex: preprocess_batch(ex, tokenizer, max_len=max_len), batched=True)
#     test_t = test.map(lambda ex: preprocess_batch(ex, tokenizer, max_len=max_len), batched=True)

#     # keep only fields we need: input_ids, attention_mask, label, text
#     wanted = ("input_ids", "attention_mask", "label", "text")
#     train_t = train_t.remove_columns([c for c in train_t.column_names if c not in wanted])
#     test_t = test_t.remove_columns([c for c in test_t.column_names if c not in wanted])

#     return train_t, test_t, tokenizer
  
  
  
  # src/data.py
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenizer(model_name="distilbert-base-uncased", model_dir=None, max_len=128):
    if model_dir:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def preprocess_batch(examples, tokenizer, max_len=128):
    """Return tokenized fields but keep original text."""
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )
    # keep original text in dataset so we can use it later for error analysis
    out["text"] = examples["text"]
    return out

def load_imdb_tokenized(model_name="distilbert-base-uncased", model_dir=None,
                        max_len=128, use_subset=True, train_subset=2000, eval_subset=1000, seed=42,
                        force_reload=False):
    """
    Returns: train_tokenized, test_tokenized, tokenizer
    Each tokenized dataset contains input_ids, attention_mask, label, text
    """
    raw = load_dataset("imdb")
    if use_subset:
        train = raw["train"].shuffle(seed=seed).select(range(train_subset))
        test = raw["test"].shuffle(seed=seed).select(range(eval_subset))
    else:
        train = raw["train"]
        test = raw["test"]

    tokenizer = get_tokenizer(model_name=model_name, model_dir=model_dir, max_len=max_len)

    # Force re-map when developing to avoid stale cache
    map_kwargs = {"batched": True}
    if force_reload:
        map_kwargs["load_from_cache_file"] = False

    train_t = train.map(lambda ex: preprocess_batch(ex, tokenizer, max_len=max_len), **map_kwargs)
    test_t = test.map(lambda ex: preprocess_batch(ex, tokenizer, max_len=max_len), **map_kwargs)

    # keep only fields we need: input_ids, attention_mask, label, text
    wanted = ("input_ids", "attention_mask", "label", "text")
    train_t = train_t.remove_columns([c for c in train_t.column_names if c not in wanted])
    test_t = test_t.remove_columns([c for c in test_t.column_names if c not in wanted])

    return train_t, test_t, tokenizer

# training dataset (no text) - safe to pass directly to model.fit
def dataset_to_tf(ds, max_len=128, batch_size=16, shuffle=False, seed=42):
    def gen():
        for ex in ds:
            features = {
                "input_ids": np.array(ex["input_ids"], dtype=np.int32),
                "attention_mask": np.array(ex["attention_mask"], dtype=np.int32),
            }
            label = np.int64(ex["label"])
            yield features, label

    output_signature = (
        {
            "input_ids": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    tf_ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
    if shuffle:
        tf_ds = tf_ds.shuffle(2048, seed=seed)
    tf_ds = tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return tf_ds

# evaluation dataset (includes text) - used only for evaluation / error analysis
def dataset_to_tf_with_text(ds, max_len=128, batch_size=16, shuffle=False, seed=42):
    def gen():
        for ex in ds:
            features = {
                "input_ids": np.array(ex["input_ids"], dtype=np.int32),
                "attention_mask": np.array(ex["attention_mask"], dtype=np.int32),
                "text": ex["text"],   # Python str
            }
            label = np.int64(ex["label"])
            yield features, label

    output_signature = (
        {
            "input_ids": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
            "text": tf.TensorSpec(shape=(), dtype=tf.string),
        },
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    tf_ds = tf.data.Dataset.from_generator(lambda: gen(), output_signature=output_signature)
    if shuffle:
        tf_ds = tf_ds.shuffle(2048, seed=seed)
    tf_ds = tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return tf_ds
