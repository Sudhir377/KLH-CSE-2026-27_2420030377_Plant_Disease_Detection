import os
import sys
import json
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import model

DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "PlantVillage",
        "raw",
        "color"
    )
)

RESULTS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "results"
    )
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nNumber of classes:", len(class_names))
print("\nClass order:")

for i, name in enumerate(class_names):
    print(i, "->", name)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# Stage 1: Train classification head
# ==============================

print("\nStarting Stage 1 training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ==============================
# Stage 2: Fine-tune EfficientNet
# ==============================

print("\nStarting Stage 2 fine-tuning...")

base_model = model.layers[3]

base_model.trainable = True

# Freeze the first layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ==============================
# Save model
# ==============================

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "plant_disease_model.keras"
)

model.save(MODEL_PATH)

# Save class names
CLASS_PATH = os.path.join(
    RESULTS_DIR,
    "class_names.json"
)

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print("\n==============================")
print("Training completed!")
print("==============================")
print("Model saved to:")
print(MODEL_PATH)

print("\nClass names saved to:")
print(CLASS_PATH)