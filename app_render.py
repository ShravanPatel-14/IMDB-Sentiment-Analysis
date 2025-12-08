# app_render.py
import logging
from typing import List

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

# ---- CONFIG ----
MAX_TEXTS = 64  # limit per request to avoid memory spikes

# ---- LOGGER ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imdb-sentiment-render")

# ---- FLASK APP ----
app = Flask(__name__)
CORS(app)

# ---- LOAD MODEL (Hugging Face pipeline) ----
logger.info("Loading Hugging Face sentiment-analysis pipeline...")
nlp = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
logger.info("Pipeline loaded.")


def pipeline_to_pred(label: str, score: float):
    """
    Convert HF label/score into:
      pred: 1 for POSITIVE, 0 for NEGATIVE
      probs: [prob_negative, prob_positive]
    So that it matches your old TensorFlow API format.
    """
    if label.upper().startswith("POS"):
        pred = 1
        probs = [1.0 - score, score]
    else:
        pred = 0
        probs = [score, 1.0 - score]
    return pred, probs


def predict_texts(texts: List[str]) -> List[dict]:
    """Run batched inference through the HF pipeline."""
    if not texts:
        return []

    # Limit for safety on free tier
    texts = texts[:MAX_TEXTS]

    results = nlp(texts)  # list of {label, score}
    out = []
    for text, r in zip(texts, results):
        label = r["label"]
        score = float(r["score"])
        pred, probs = pipeline_to_pred(label, score)
        out.append({
            "text": text,
            "pred": int(pred),
            "probs": probs,
        })
    return out


# ---- ROUTES ----
@app.route("/", methods=["GET"])
def index():
    # Simple JSON root
    return jsonify({"status": "ok", "service": "imdb-sentiment-render"})


@app.route("/ui", methods=["GET"])
def ui():
    # Reuse your existing templates/index.html
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    ok = nlp is not None
    return jsonify({"ready": ok}), (200 if ok else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if nlp is None:
        return jsonify({"error": "model not loaded"}), 503

    if not request.is_json:
        return jsonify({"error": "Expected application/json"}), 400

    body = request.get_json()
    texts = body.get("texts") or body.get("inputs") or body.get("data")
    if texts is None:
        return jsonify({"error": "JSON must contain 'texts' (a list of strings)"}), 400
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400

    cleaned = [str(t).strip() for t in texts if str(t).strip()]

    if not cleaned:
        return jsonify({"predictions": []}), 200

    try:
        preds = predict_texts(cleaned)
        return jsonify({"predictions": preds}), 200
    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": "inference failed", "details": str(e)}), 500


if __name__ == "__main__":
    # For local testing (not used on Render: there we use gunicorn)
    app.run(host="0.0.0.0", port=5000, debug=False)
