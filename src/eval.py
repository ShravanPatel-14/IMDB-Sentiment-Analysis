# # src/eval.py
# import os
# import json
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# import seaborn as sns
# import tensorflow as tf

# def evaluate_model_and_save(model, tokenizer, tf_dataset, run_dir, max_len=128, top_k_errors=50):
#     """
#     Runs batched inference on tf_dataset (yields (features,label)),
#     computes metrics, saves metrics.json, confusion matrix PNG and top_errors CSV (with original text).
#     """
#     os.makedirs(run_dir, exist_ok=True)
#     y_true = []
#     y_pred = []
#     raw_texts = []
#     probs_list = []

#     # iterate dataset: tf_dataset yields batched ({'input_ids','attention_mask','text'}, label)
#     for batch in tf_dataset:
#         features, labels = batch
#         # features['text'] is a tf.Tensor of shape (batch,) dtype=string
#         # decode to Python strings:
#         try:
#             text_batch = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else t for t in features["text"].numpy()]
#         except Exception:
#             # sometimes tf returns numpy.str_ objects; convert to str
#             text_batch = [str(t) for t in features["text"].numpy()]

#         # run model
#         logits = model(features, training=False)[0]
#         probs = tf.nn.softmax(logits, axis=-1).numpy()
#         preds = np.argmax(probs, axis=-1).tolist()

#         y_true.extend(labels.numpy().tolist())
#         y_pred.extend(preds)
#         probs_list.extend(probs.tolist())
#         raw_texts.extend(text_batch)
#     # Metrics
#     acc = float(accuracy_score(y_true, y_pred))
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
#     metrics = {
#         "accuracy": acc,
#         "precision": float(precision),
#         "recall": float(recall),
#         "f1": float(f1),
#         "n_samples": len(y_true)
#     }

#     # Save metrics.json
#     metrics_path = os.path.join(run_dir, "metrics.json")
#     with open(metrics_path, "w") as f:
#         json.dump(metrics, f, indent=2)

#     # Confusion matrix plot
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     cm_path = os.path.join(run_dir, "confusion_matrix.png")
#     plt.tight_layout()
#     plt.savefig(cm_path)
#     plt.close()

#     # Top errors CSV: collect false positives and false negatives with probs
#     # false positive: true=0 pred=1 ; false negative: true=1 pred=0
#     errors = []
#     for i, (t,p,probs) in enumerate(zip(y_true, y_pred, probs_list)):
#         if t != p:
#             errors.append({
#                 "index": i,
#                 "true": int(t),
#                 "pred": int(p),
#                 "prob_true": float(probs[int(t)]),
#                 "prob_pred": float(probs[int(p)])
#             })
#     # Sort by model confidence in wrong class descending (hard errors first)
#     errors_sorted = sorted(errors, key=lambda x: x["prob_pred"], reverse=True)
#     top_errors = errors_sorted[:top_k_errors]

#     csv_path = os.path.join(run_dir, "top_errors.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=["index","true","pred","prob_true","prob_pred"])
#         writer.writeheader()
#         for row in top_errors:
#             writer.writerow(row)

#     return {
#         "metrics_path": metrics_path,
#         "confusion_matrix": cm_path,
#         "errors_csv": csv_path,
#         "metrics": metrics
#     }


# src/eval.py
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import tensorflow as tf

def evaluate_model_and_save(model, tokenizer, tf_dataset_with_text, run_dir, max_len=128, top_k_errors=50):
    """
    Runs batched inference on tf_dataset_with_text (yields (features,label)),
    computes metrics, saves metrics.json, confusion matrix PNG and top_errors CSV (with original text).
    """
    os.makedirs(run_dir, exist_ok=True)
    y_true = []
    y_pred = []
    raw_texts = []
    probs_list = []

    for batch in tf_dataset_with_text:
        features, labels = batch

        # decode text batch to Python strings
        try:
            text_batch = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else t for t in features["text"].numpy()]
        except Exception:
            text_batch = [str(t) for t in features["text"].numpy()]

        # pass only model inputs (strip 'text')
        model_inputs = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        logits = model(model_inputs, training=False)[0]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        preds = np.argmax(probs, axis=-1).tolist()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds)
        probs_list.extend(probs.tolist())
        raw_texts.extend(text_batch)

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_samples": len(y_true)
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Top errors with text
    errors = []
    for i, (t_label, p_label, probs, txt) in enumerate(zip(y_true, y_pred, probs_list, raw_texts)):
        if t_label != p_label:
            errors.append({
                "index": i,
                "text": txt,
                "true": int(t_label),
                "pred": int(p_label),
                "prob_true": float(probs[int(t_label)]),
                "prob_pred": float(probs[int(p_label)])
            })
    errors_sorted = sorted(errors, key=lambda x: x["prob_pred"], reverse=True)
    top_errors = errors_sorted[:top_k_errors]

    csv_path = os.path.join(run_dir, "top_errors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index","text","true","pred","prob_true","prob_pred"])
        writer.writeheader()
        for row in top_errors:
            writer.writerow(row)

    return {
        "metrics_path": metrics_path,
        "confusion_matrix": cm_path,
        "errors_csv": csv_path,
        "metrics": metrics
    }
