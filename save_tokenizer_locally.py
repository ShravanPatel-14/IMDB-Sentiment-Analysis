from transformers import AutoTokenizer
MODEL_DIR = "./tf_distilbert_imdb"
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.save_pretrained(MODEL_DIR)
print("Saved tokenizer files to", MODEL_DIR)
