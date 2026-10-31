"""
PlantCare AI
Model Evaluation
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "data" / "PlantVillage"
MODEL_PATH = PROJECT_DIR / "models" / "plant_disease_efficientnet.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("=" * 70)
print("PLANTCARE AI - MODEL EVALUATION")
print("=" * 70)

print("\nNumber of classes:", len(class_names))

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# ============================================================
# COLLECT DATASET
# ============================================================

class_folders = sorted(
    folder
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
    and folder.name.lower() != "plantvillage"
)

class_to_index = {
    name: index
    for index, name in enumerate(
        [folder.name for folder in class_folders]
    )
}

image_paths = []
labels = []


print("\nCollecting image paths...")

for folder in class_folders:

    files = sorted(
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    for file in files:
        image_paths.append(str(file))
        labels.append(class_to_index[folder.name])


image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)


print("Total images:", len(image_paths))


# ============================================================
# SAME 80 / 10 / 10 SPLIT USED DURING TRAINING
# ============================================================

train_paths, remaining_paths, train_labels, remaining_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_labels,
    )
)


print("\nDataset split:")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels,
    )
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_ds = test_ds.batch(
    BATCH_SIZE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# BASIC MODEL EVALUATION
# ============================================================

print("\nEvaluating test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels_batch in test_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(
        labels_batch.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)


plt.figure(
    figsize=(15, 12)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.title(
    "Plant Disease Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    rotation=90
)

plt.yticks(
    rotation=0
)

plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()


# ============================================================
# FINISHED
# ============================================================

metrics_payload = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "model": "EfficientNetB0 transfer learning",
    "dataset": "PlantVillage project folder",
    "class_count": int(len(class_names)),
    "total_images": int(len(image_paths)),
    "train_images": int(len(train_paths)),
    "validation_images": int(len(val_paths)),
    "test_images": int(len(test_paths)),
    "test_loss": float(test_loss),
    "accuracy_percent": round(float(accuracy) * 100, 4),
    "weighted_precision_percent": round(float(precision) * 100, 4),
    "weighted_recall_percent": round(float(recall) * 100, 4),
    "weighted_f1_percent": round(float(f1) * 100, 4),
    "class_names": class_names,
    "note": "Held-out split metrics only; not an independent real-world field evaluation."
}
with open(REPORT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as report_file:
    json.dump(metrics_payload, report_file, indent=2)

print(f"\nSaved metrics: {REPORT_DIR / 'evaluation_metrics.json'}")
print(f"Saved confusion matrix: {REPORT_DIR / 'confusion_matrix.png'}")

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)