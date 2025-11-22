# # src/train.py
# import os
# import time

# def train_and_save(model, train_tf, val_tf, output_dir="./runs", epochs=1, callbacks=None):
#     run_id = str(int(time.time()))
#     run_dir = os.path.join(output_dir, run_id)
#     os.makedirs(run_dir, exist_ok=True)

#     history = model.fit(
#         train_tf,
#         validation_data=val_tf,
#         epochs=epochs,
#         callbacks=callbacks or [],
#         verbose=1
#     )

#     # try to save best weights if callback used, else save pretrained HF style
#     try:
#         model.save_pretrained(run_dir)
#     except Exception:
#         # fallback: save weights only
#         weights_path = os.path.join(run_dir, "weights.h5")
#         model.save_weights(weights_path)

#     return history, run_dir



# src/train.py
import os
import time

def train_and_save(model, train_tf, val_tf, output_dir="./runs", epochs=1, callbacks=None):
    run_id = str(int(time.time()))
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1
    )

    # Save model in HF-compatible format if possible; otherwise save weights
    try:
        model.save_pretrained(run_dir)
    except Exception:
        weights_path = os.path.join(run_dir, "weights.h5")
        model.save_weights(weights_path)

    return history, run_dir
