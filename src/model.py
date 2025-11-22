
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

def build_model(model_name_or_dir="distilbert-base-uncased", num_labels=2, learning_rate=2e-5):
    # Load TF model (will download if not present)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_dir, num_labels=num_labels,from_pt=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
