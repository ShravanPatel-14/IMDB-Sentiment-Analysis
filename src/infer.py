# # src/infer.py
# import numpy as np
# import tensorflow as tf

# def predict_texts(model, tokenizer, texts, max_len=128):
#     toks = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
#     logits = model(toks)[0]
#     probs = tf.nn.softmax(logits, axis=-1).numpy()
#     preds = np.argmax(probs, axis=-1)
#     results = []
#     for t, p, pr in zip(texts, preds, probs.tolist()):
#         results.append({"text": t, "pred": int(p), "probs": pr})
#     return results


# src/infer.py
import numpy as np
import tensorflow as tf

def predict_texts(model, tokenizer, texts, max_len=128):
    toks = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
    logits = model(toks)[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    results = []
    for t, p, pr in zip(texts, preds, probs.tolist()):
        results.append({"text": t, "pred": int(p), "probs": pr})
    return results
