import os
import glob
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EXTRA_SITE_PACKAGES = r"C:\tfpkgs"
if os.path.isdir(EXTRA_SITE_PACKAGES) and EXTRA_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, EXTRA_SITE_PACKAGES)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
except Exception as exc:
    raise SystemExit(
        "TensorFlow is unavailable or broken in this environment. "
        "Reinstall TensorFlow and retry.\n"
        f"Original error: {exc}"
    )

STEP_SIZE = 20
SENSOR_NUM = 6
NUM_CLASSES = 7

DATASET_GLOB = "./Example Datasets/Example Datasets/**/*.csv"
MODEL_FILE = "./model_x.keras"

LABEL_MAP = {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "FALL": 4, "LYI": 5, "RA": 6}
CLASS_NAMES = {0: "STD", 1: "WAL", 2: "JOG", 3: "JUM", 4: "FALL", 5: "LYI", 6: "RA"}
CLASS_LIST = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]


class LiveTrainingPlot(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.enabled = True

    def on_train_begin(self, logs=None):
        try:
            plt.ion()
            self.fig, (self.ax_acc, self.ax_loss) = plt.subplots(2, 1, figsize=(9, 7))
            self.acc_train_line, = self.ax_acc.plot([], [], label="Train Acc", color="tab:blue")
            self.acc_val_line, = self.ax_acc.plot([], [], label="Val Acc", color="tab:orange")
            self.ax_acc.set_title("Training Accuracy")
            self.ax_acc.set_xlabel("Epoch")
            self.ax_acc.set_ylabel("Accuracy")
            self.ax_acc.grid(alpha=0.3)
            self.ax_acc.legend()

            self.loss_train_line, = self.ax_loss.plot([], [], label="Train Loss", color="tab:green")
            self.loss_val_line, = self.ax_loss.plot([], [], label="Val Loss", color="tab:red")
            self.ax_loss.set_title("Training Loss")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.grid(alpha=0.3)
            self.ax_loss.legend()

            self.fig.tight_layout()
            plt.show(block=False)
        except Exception:
            self.enabled = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return

        logs = logs or {}
        self.epochs.append(epoch + 1)
        self.train_acc.append(float(logs.get("accuracy", 0.0)))
        self.val_acc.append(float(logs.get("val_accuracy", 0.0)))
        self.train_loss.append(float(logs.get("loss", 0.0)))
        self.val_loss.append(float(logs.get("val_loss", 0.0)))

        self.acc_train_line.set_data(self.epochs, self.train_acc)
        self.acc_val_line.set_data(self.epochs, self.val_acc)
        self.ax_acc.set_xlim(1, max(2, self.epochs[-1]))
        self.ax_acc.set_ylim(0.0, 1.05)

        self.loss_train_line.set_data(self.epochs, self.train_loss)
        self.loss_val_line.set_data(self.epochs, self.val_loss)
        self.ax_loss.set_xlim(1, max(2, self.epochs[-1]))
        max_loss = max(max(self.train_loss), max(self.val_loss))
        self.ax_loss.set_ylim(0.0, max(1.0, max_loss * 1.1))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def on_train_end(self, logs=None):
        if not self.enabled:
            return
        plt.ioff()


def build_windows(features, labels, step_size):
    samples = []
    targets = []
    for i in range(len(features) - step_size):
        samples.append(features[i : i + step_size])
        label_window = labels[i : i + step_size]
        targets.append(max(label_window, key=label_window.count))
    return np.array(samples).reshape(-1, step_size, SENSOR_NUM), np.array(targets)


def save_confusion_matrices(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{prefix}_confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_per_class_metrics(y_true, y_pred, prefix):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    x = np.arange(NUM_CLASSES)
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width=width, label="Precision")
    plt.bar(x, recall, width=width, label="Recall")
    plt.bar(x + width, f1, width=width, label="F1")
    plt.xticks(x, CLASS_LIST)
    plt.ylim(0, 1.05)
    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(f"{prefix}_per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(CLASS_LIST, support, color="tab:purple")
    plt.title("Class Support")
    plt.ylabel("Samples")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{prefix}_class_support.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_class_distribution(y_true, y_pred, prefix):
    true_counts = np.bincount(y_true, minlength=NUM_CLASSES)
    pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
    x = np.arange(NUM_CLASSES)
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, true_counts, width=width, label="Actual")
    plt.bar(x + width / 2, pred_counts, width=width, label="Predicted")
    plt.xticks(x, CLASS_LIST)
    plt.title("Actual vs Predicted Class Distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(f"{prefix}_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_roc_pr_curves(y_true, y_prob, prefix):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_LIST):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("One-vs-Rest ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.savefig(f"{prefix}_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    ap_scores = []
    for i, class_name in enumerate(CLASS_LIST):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ap_scores.append(ap)
        plt.plot(r, p, label=f"{class_name} (AP={ap:.3f})")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower left")
    plt.savefig(f"{prefix}_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(CLASS_LIST, ap_scores, color="tab:cyan")
    plt.ylim(0, 1.05)
    plt.title("Average Precision by Class")
    plt.ylabel("AP")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{prefix}_average_precision.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_confidence_error_analysis(y_true, y_pred, y_prob, prefix):
    confidence = y_prob[np.arange(len(y_pred)), y_pred]
    correct_mask = y_true == y_pred

    plt.figure(figsize=(9, 5))
    plt.hist(confidence[correct_mask], bins=30, alpha=0.7, label="Correct", color="tab:green")
    plt.hist(confidence[~correct_mask], bins=30, alpha=0.7, label="Wrong", color="tab:red")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted-Class Probability")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"{prefix}_confidence_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    error_counts = np.bincount(y_true[~correct_mask], minlength=NUM_CLASSES)
    plt.figure(figsize=(9, 4))
    plt.bar(CLASS_LIST, error_counts, color="tab:red")
    plt.title("Errors by True Class")
    plt.ylabel("Misclassified Samples")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{prefix}_errors_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()


csv_files = glob.glob(DATASET_GLOB, recursive=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found using: {DATASET_GLOB}")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
dataset = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label"]].copy()
dataset["label"] = dataset["label"].map(LABEL_MAP)

x = np.array(dataset.drop(["label"], axis=1))
y = np.array(dataset["label"])
windowed_x, windowed_y = build_windows(x, y.tolist(), STEP_SIZE)

x_train, x_test, y_train, y_test = train_test_split(
    windowed_x, windowed_y, test_size=0.3, random_state=42, stratify=windowed_y
)

model = Sequential()
model.add(keras.layers.Flatten(input_shape=(STEP_SIZE, SENSOR_NUM)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=30,
    validation_split=0.1,
    callbacks=[LiveTrainingPlot()],
)

model.save(MODEL_FILE)

pred = model.predict(x_test)
results = np.argmax(pred, axis=1)

for i in range(min(50, len(x_test))):
    if CLASS_NAMES[y_test[i]] == CLASS_NAMES[results[i]]:
        print("prediction:", CLASS_NAMES[results[i]], "actual:", CLASS_NAMES[y_test[i]], "Correct!")
    else:
        print("prediction:", CLASS_NAMES[results[i]], "actual:", CLASS_NAMES[y_test[i]], "Wrong!")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("training_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
plt.close()

save_confusion_matrices(y_test, results, "training")
save_per_class_metrics(y_test, results, "training")
save_class_distribution(y_test, results, "training")
save_roc_pr_curves(y_test, pred, "training")
save_confidence_error_analysis(y_test, results, pred, "training")

print("\nClassification Report:\n")
print(classification_report(y_test, results, target_names=CLASS_LIST))
