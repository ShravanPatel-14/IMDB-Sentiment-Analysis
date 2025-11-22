
import os
import logging
from typing import List
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ---- CONFIG (match your training script) ----
MODEL_DIR = os.environ.get("MODEL_DIR", "./tf_distilbert_imdb")  # same as OUTPUT_DIR in your training script
MAX_LEN = 128
BATCH_SIZE = 16
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ---- LOGGER ----
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("tf-imdb-api")

# ---- FLASK APP ----
app = Flask(__name__)
CORS(app)

# ---- GLOBALS (loaded at startup) ----
model = None
tokenizer = None

def load_model_and_tokenizer(model_dir: str, hf_fallback: str = "distilbert-base-uncased"):
    """
    Robust loader:
      - tries to load tokenizer from model_dir
      - if tokenizer files missing, loads tokenizer from HF `hf_fallback` and saves it into model_dir
      - then loads TF model from model_dir (or falls back to hf_fallback if needed)
    """
    global model, tokenizer

    logger.info("Attempting to load tokenizer from: %s", model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        logger.info("Loaded tokenizer from %s", model_dir)
    except Exception as e:
        logger.warning("Could not load tokenizer from %s: %s", model_dir, str(e))
        logger.info("Falling back to tokenizer from Hugging Face hub: %s", hf_fallback)
        tokenizer = AutoTokenizer.from_pretrained(hf_fallback, use_fast=True)
        # ensure local dir exists and save tokenizer for future runs
        try:
            os.makedirs(model_dir, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            logger.info("Saved tokenizer files to %s", model_dir)
        except Exception as save_err:
            logger.warning("Failed to save tokenizer to %s: %s", model_dir, str(save_err))

    # Now load model (prefer local model_dir; if that fails, fallback to hf_fallback)
    logger.info("Attempting to load TF model from: %s", model_dir)
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        logger.info("Loaded TF model from %s", model_dir)
    except Exception as e:
        logger.warning("Could not load TF model from %s: %s", model_dir, str(e))
        logger.info("Falling back to HF model name: %s", hf_fallback)
        model = TFAutoModelForSequenceClassification.from_pretrained(hf_fallback)
        logger.info("Loaded TF model from HF hub: %s", hf_fallback)


def batchify(iterable: List[str], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def predict_texts(texts: List[str]) -> List[dict]:
    """
    Batched inference. Returns list of {"text":..., "pred":int, "probs":[...]}
    """
    results = []
    if not texts:
        return results

    for sub in batchify(texts, BATCH_SIZE):
        toks = tokenizer(
            sub,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="tf"
        )
        logits = model(toks, training=False)[0]              # (batch, num_labels)
        probs = tf.nn.softmax(logits, axis=-1).numpy()       # (batch, num_labels)
        preds = np.argmax(probs, axis=-1).tolist()
        for text, pred, prob in zip(sub, preds, probs.tolist()):
            results.append({"text": text, "pred": int(pred), "probs": prob})
    return results

# ---- Try to load model/tokenizer at import time so Gunicorn workers have them.
# This runs when the module is imported (works with `gunicorn app:app`).
try:
    logger.info("Import-time loading of model/tokenizer from: %s", MODEL_DIR)
    load_model_and_tokenizer(MODEL_DIR)
except Exception as e:
    logger.exception("Failed to load model/tokenizer at import time: %s", e)
    # model and tokenizer remain None; /health will show not ready and /predict will return 503

# ---- ROUTES ----
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "model_dir": MODEL_DIR})

@app.route("/health", methods=["GET"])
def health():
    ok = model is not None and tokenizer is not None
    return jsonify({"ready": ok}), (200 if ok else 503)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "model not loaded"}), 503

    if not request.is_json:
        return jsonify({"error": "Expected application/json"}), 400

    body = request.get_json()
    texts = body.get("texts") or body.get("inputs") or body.get("data")
    if texts is None:
        return jsonify({"error": "JSON must contain 'texts' (a list of strings)"}), 400
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    # sanitize & coerce
    cleaned = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        cleaned.append(t.strip())

    # optional limits to avoid DoS
    if len(cleaned) > 512:
        return jsonify({"error": "Too many texts in one request; limit is 512"}), 400

    try:
        preds = predict_texts(cleaned)
        return jsonify({"predictions": preds}), 200
    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": "inference failed", "details": str(e)}), 500

# serve frontend
@app.route("/ui", methods=["GET"])
def ui():
    return render_template("index.html")


# ---- STARTUP (only used when running `python app.py`) ----
if __name__ == "__main__":
    if not os.path.isdir(MODEL_DIR):
        logger.error(
            "MODEL_DIR does not exist: %s. Save your model with model.save_pretrained(OUTPUT_DIR) first.",
            MODEL_DIR,
        )
        raise SystemExit(1)

    # If you run directly with `python app.py`, ensure the model is loaded (import-time loader also runs).
    if model is None or tokenizer is None:
        load_model_and_tokenizer(MODEL_DIR)

    # dev server (not for production)
    app.run(host=HOST, port=PORT, debug=False)
