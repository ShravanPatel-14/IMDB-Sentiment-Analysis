
import os
import json
import csv
import matplotlib.pyplot as plt

def save_history_json(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return path

def save_history_csv(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "history.csv")
    keys = list(history.keys())
    length = len(history[keys[0]]) if keys else 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(length):
            row = [history[k][i] for k in keys]
            writer.writerow(row)
    return csv_path

def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    acc_path = None
    loss_path = None

    if acc:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, acc, marker='o', label="train_acc")
        if val_acc:
            plt.plot(epochs, val_acc, marker='o', label="val_acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        acc_path = os.path.join(out_dir, "accuracy.png")
        plt.tight_layout()
        plt.savefig(acc_path)
        plt.close()

    if loss:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, loss, marker='o', label="train_loss")
        if val_loss:
            plt.plot(epochs, val_loss, marker='o', label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        loss_path = os.path.join(out_dir, "loss.png")
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()

    return {"accuracy_png": acc_path, "loss_png": loss_path}
