import glob
import os
import sys

EXTRA_SITE_PACKAGES = r"C:\tfpkgs"
if os.path.isdir(EXTRA_SITE_PACKAGES) and EXTRA_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, EXTRA_SITE_PACKAGES)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(
        "TensorFlow is unavailable or broken in this environment. "
        "Reinstall TensorFlow and retry.\n"
        f"Original error: {exc}"
    )

STEP_SIZE = 20
SENSOR_NUM = 6

DATASET_GLOB = "./Example Datasets/Example Datasets/**/*.csv"
MODEL_DIR = "./model_x"
MODEL_FILE = "./model_x.keras"

LABEL_MAP = {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "FALL": 4, "LYI": 5, "RA": 6}
CLASS_NAMES = ["STD", "WAL", "JOG", "JUM", "FALL", "LYI", "RA"]
NUM_CLASSES = len(CLASS_NAMES)


def build_windows(features, labels, step_size):
    samples = []
    targets = []
    for i in range(len(features) - step_size):
        samples.append(features[i : i + step_size])
        label_window = labels[i : i + step_size]
        targets.append(max(label_window, key=label_window.count))
    return np.array(samples).reshape(-1, step_size, SENSOR_NUM), np.array(targets)


def load_inference_model():
    if os.path.isfile(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    try:
        return tf.keras.models.load_model(MODEL_DIR)
    except ValueError:
        try:
            layer = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
            inputs = tf.keras.Input(shape=(STEP_SIZE, SENSOR_NUM), dtype=tf.float32)
            outputs = layer(inputs)
            if isinstance(outputs, dict):
                outputs = next(iter(outputs.values()))
            return tf.keras.Model(inputs, outputs)
        except Exception as exc:
            raise SystemExit(
                "Could not load model from './model_x' or './model_x.keras'. "
                "Run Model_Training.py first to generate './model_x.keras'.\n"
                f"Original error: {exc}"
            )


def show_live_test_progress(y_true, y_pred):
    try:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    except Exception:
        return

    ax_acc = axes[0, 0]
    ax_bar = axes[0, 1]
    ax_fall = axes[1, 0]
    ax_cm = axes[1, 1]

    running_x = []
    running_acc = []
    running_fall_precision = []
    running_fall_recall = []
    predicted_counts = np.zeros(NUM_CLASSES, dtype=int)
    cm_running = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    correct = 0

    (acc_line,) = ax_acc.plot([], [], color="tab:blue", label="Running Accuracy")
    ax_acc.set_title("Testing Progress")
    ax_acc.set_xlabel("Sample")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    bars = ax_bar.bar(CLASS_NAMES, predicted_counts, color="tab:green")
    ax_bar.set_title("Predicted Class Counts")
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Count")
    ax_bar.grid(axis="y", alpha=0.3)

    (fall_p_line,) = ax_fall.plot([], [], color="tab:orange", label="FALL Precision")
    (fall_r_line,) = ax_fall.plot([], [], color="tab:red", label="FALL Recall")
    ax_fall.set_title("FALL Class Trend")
    ax_fall.set_xlabel("Sample")
    ax_fall.set_ylabel("Score")
    ax_fall.set_ylim(0.0, 1.05)
    ax_fall.grid(alpha=0.3)
    ax_fall.legend()

    cm_img = ax_cm.imshow(cm_running, cmap="Blues", vmin=0)
    ax_cm.set_title("Running Confusion Matrix")
    ax_cm.set_xticks(np.arange(NUM_CLASSES))
    ax_cm.set_yticks(np.arange(NUM_CLASSES))
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax_cm.set_yticklabels(CLASS_NAMES)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    fig.colorbar(cm_img, ax=ax_cm, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show(block=False)

    for i, (actual, pred) in enumerate(zip(y_true, y_pred), start=1):
        if actual == pred:
            correct += 1
        predicted_counts[pred] += 1
        cm_running[actual, pred] += 1
        running_x.append(i)
        running_acc.append(correct / i)
        fall_tp = cm_running[4, 4]
        fall_fp = cm_running[:, 4].sum() - fall_tp
        fall_fn = cm_running[4, :].sum() - fall_tp
        fall_precision = fall_tp / max(1, (fall_tp + fall_fp))
        fall_recall = fall_tp / max(1, (fall_tp + fall_fn))
        running_fall_precision.append(fall_precision)
        running_fall_recall.append(fall_recall)

        if i % 20 == 0 or i == len(y_true):
            acc_line.set_data(running_x, running_acc)
            ax_acc.set_xlim(1, max(2, i))

            for idx, bar in enumerate(bars):
                bar.set_height(predicted_counts[idx])
            ax_bar.set_ylim(0, max(1, int(predicted_counts.max() * 1.2)))

            fall_p_line.set_data(running_x, running_fall_precision)
            fall_r_line.set_data(running_x, running_fall_recall)
            ax_fall.set_xlim(1, max(2, i))

            cm_img.set_data(cm_running)
            cm_img.set_clim(0, max(1, cm_running.max()))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

    plt.ioff()


def save_confusion_matrices(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    plt.figure(figsize=(8, 6))
    sn.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
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
    plt.xticks(x, CLASS_NAMES)
    plt.ylim(0, 1.05)
    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(f"{prefix}_per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(CLASS_NAMES, support, color="tab:purple")
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
    plt.xticks(x, CLASS_NAMES)
    plt.title("Actual vs Predicted Class Distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(f"{prefix}_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_roc_pr_curves(y_true, y_prob, prefix):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
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
    for i, class_name in enumerate(CLASS_NAMES):
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
    plt.bar(CLASS_NAMES, ap_scores, color="tab:cyan")
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
    plt.bar(CLASS_NAMES, error_counts, color="tab:red")
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

model = load_inference_model()
model.summary()

pred = model.predict(windowed_x)
results = np.argmax(pred, axis=1)

show_live_test_progress(windowed_y, results)

correct = int((results == windowed_y).sum())
total = len(windowed_y)
print("success rate:", (correct / total) * 100.0)

for i in range(min(50, total)):
    predicted = CLASS_NAMES[results[i]]
    actual = CLASS_NAMES[windowed_y[i]]
    status = "Correct!" if predicted == actual else "Wrong!"
    print("prediction:", predicted, "actual:", actual, status)

cm = confusion_matrix(windowed_y, results)
print(cm)

print(classification_report(windowed_y, results, target_names=CLASS_NAMES))
save_confusion_matrices(windowed_y, results, "testing")
save_per_class_metrics(windowed_y, results, "testing")
save_class_distribution(windowed_y, results, "testing")
save_roc_pr_curves(windowed_y, pred, "testing")
save_confidence_error_analysis(windowed_y, results, pred, "testing")
