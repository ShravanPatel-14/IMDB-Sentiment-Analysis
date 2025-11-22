
from src.data import load_imdb_tokenized, dataset_to_tf, dataset_to_tf_with_text
from src.model import build_model
from src.train import train_and_save
from src.infer import predict_texts
from src.utils_logging import save_history_json, save_history_csv, plot_history
from src.eval import evaluate_model_and_save

# quick config for smoke test
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 8
EPOCHS = 1
USE_SUBSET = True
TRAIN_SUBSET = 500
EVAL_SUBSET = 200
FORCE_RELOAD = True   # force re-tokenization (set False later)

# 1) load tokenized datasets (tokenizer included)
train_t, test_t, tokenizer = load_imdb_tokenized(
    model_name=MODEL_NAME,
    model_dir=None,
    max_len=MAX_LEN,
    use_subset=USE_SUBSET,
    train_subset=TRAIN_SUBSET,
    eval_subset=EVAL_SUBSET,
    seed=42,
    force_reload=FORCE_RELOAD
)

# 2) convert to tf.data
train_tf = dataset_to_tf(train_t, max_len=MAX_LEN, batch_size=BATCH, shuffle=True)
# validation for fit should NOT include text -> use dataset_to_tf
val_tf_for_fit = dataset_to_tf(test_t, max_len=MAX_LEN, batch_size=BATCH, shuffle=False)
# evaluation dataset including text for error analysis
test_tf_with_text = dataset_to_tf_with_text(test_t, max_len=MAX_LEN, batch_size=BATCH, shuffle=False)

# 3) build model
model = build_model(MODEL_NAME, num_labels=2, learning_rate=2e-5)

# 4) train and save
history, run_dir = train_and_save(model, train_tf, val_tf_for_fit, output_dir="./runs", epochs=EPOCHS)

# 5) save history & plots
history_dict = history.history
from src.utils_logging import save_history_json, save_history_csv, plot_history
save_history_json(history_dict, run_dir)
save_history_csv(history_dict, run_dir)
plot_paths = plot_history(history_dict, run_dir)
print("Saved history and plots to:", run_dir)
print(plot_paths)

# 6) quick inference
samples = ["I loved this movie!", "Terrible movie — waste of time."]
results = predict_texts(model, tokenizer, samples, max_len=MAX_LEN)
print("Sample predictions:", results)

# 7) evaluate on test set and save metrics/plots/top errors (with text)
eval_out = evaluate_model_and_save(model, tokenizer, test_tf_with_text, run_dir, max_len=MAX_LEN, top_k_errors=200)
print("Evaluation saved. Metrics:", eval_out["metrics"])
