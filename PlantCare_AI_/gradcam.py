"""
PlantCare AI
Grad-CAM Explainability

Shows which part of a leaf influenced the disease prediction.
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "plant_disease_efficientnet.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "class_names.json"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 60)
print("PLANTCARE AI - GRAD-CAM")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)

print("Model loaded successfully.")


# ============================================================
# 3. FIND EFFICIENTNET MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            base_model = layer
            break


if base_model is None:

    raise RuntimeError(
        "EfficientNet model could not be found."
    )


print(
    "EfficientNet:",
    base_model.name
)


# ============================================================
# 4. FIND A SUITABLE LAST FEATURE LAYER
# ============================================================

target_layer = None

for layer in reversed(base_model.layers):

    try:

        output_shape = layer.output.shape

        if len(output_shape) == 4:

            target_layer = layer
            break

    except Exception:

        continue


if target_layer is None:

    raise RuntimeError(
        "Could not find a suitable Grad-CAM layer."
    )


print(
    "Grad-CAM layer:",
    target_layer.name
)


# ============================================================
# 5. FIND CLASSIFICATION HEAD
# ============================================================

gap_layer = None
dropout_layer = None
prediction_layer = None


for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.GlobalAveragePooling2D
    ):

        gap_layer = layer

    elif isinstance(
        layer,
        tf.keras.layers.Dropout
    ):

        dropout_layer = layer

    elif (
        isinstance(
            layer,
            tf.keras.layers.Dense
        )
        and layer.units == len(class_names)
    ):

        prediction_layer = layer


if gap_layer is None:
    raise RuntimeError(
        "GlobalAveragePooling layer not found."
    )

if prediction_layer is None:
    raise RuntimeError(
        "Prediction layer not found."
    )


# ============================================================
# 6. CREATE GRAD-CAM MODEL
# ============================================================

gradcam_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        target_layer.output,
        base_model.output
    ]
)


# ============================================================
# 7. SELECT LEAF IMAGE
# ============================================================

print("\nSelect a leaf image...")

root = Tk()
root.withdraw()

image_path = askopenfilename(
    title="Select Plant Leaf Image",
    filetypes=[
        (
            "Image files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        ),
        (
            "All files",
            "*.*"
        )
    ]
)

root.destroy()


if not image_path:

    print("No image selected.")
    raise SystemExit


print("\nSelected image:")
print(image_path)


# ============================================================
# 8. LOAD AND PREPROCESS IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(
    image
)

input_image = np.expand_dims(
    image_array,
    axis=0
)

input_image = tf.cast(
    input_image,
    tf.float32
)


# ============================================================
# 9. GENERATE GRAD-CAM
# ============================================================

with tf.GradientTape() as tape:

    feature_maps, base_output = gradcam_model(
        input_image,
        training=False
    )

    # Classification head
    pooled_output = gap_layer(
        base_output
    )

    if dropout_layer is not None:

        pooled_output = dropout_layer(
            pooled_output,
            training=False
        )

    predictions = prediction_layer(
        pooled_output
    )

    predicted_index = tf.argmax(
        predictions[0]
    )

    predicted_score = predictions[
        0,
        predicted_index
    ]


# ============================================================
# 10. CALCULATE GRADIENTS
# ============================================================

gradients = tape.gradient(
    predicted_score,
    feature_maps
)


if gradients is None:

    raise RuntimeError(
        "Gradients could not be calculated."
    )


# Average gradients across image dimensions

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)


feature_maps = feature_maps[0]

pooled_gradients = pooled_gradients[0]


# Weight each feature map

weighted_features = (
    feature_maps
    * pooled_gradients
)


heatmap = tf.reduce_sum(
    weighted_features,
    axis=-1
)


# ============================================================
# 11. NORMALIZE HEATMAP
# ============================================================

heatmap = tf.maximum(
    heatmap,
    0
)

maximum = tf.reduce_max(
    heatmap
)

heatmap = heatmap / (
    maximum + 1e-8
)

heatmap = heatmap.numpy()


# ============================================================
# 12. PREDICTION INFORMATION
# ============================================================

predicted_index = int(
    predicted_index.numpy()
)

confidence = float(
    predictions[
        0,
        predicted_index
    ].numpy()
) * 100

predicted_class = class_names[
    predicted_index
]


print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

print(
    "Disease:",
    predicted_class
)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    "Grad-CAM layer:",
    target_layer.name
)


# ============================================================
# 13. DISPLAY ORIGINAL + GRAD-CAM
# ============================================================

plt.figure(
    figsize=(12, 5)
)


# Original image

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    image
)

plt.title(
    f"Original Leaf\n"
    f"{predicted_class}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")


# Grad-CAM

plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    image
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.title(
    "Grad-CAM\n"
    "Important Region"
)

plt.axis("off")


plt.tight_layout()

plt.show()


print("\nGrad-CAM completed successfully.")