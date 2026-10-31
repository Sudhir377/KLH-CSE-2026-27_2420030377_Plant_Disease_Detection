import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# ============================================================
# PLANTCARE AI - MAIN APPLICATION
# ============================================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide"
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_efficientnet.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (224, 224)


# ------------------------------------------------------------
# Disease information
# ------------------------------------------------------------

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bell Bacterial Spot",
        "type": "Bacterial disease",
        "treatment": [
            "Remove severely infected leaves.",
            "Avoid overhead watering.",
            "Improve air circulation.",
            "Follow locally recommended bacterial disease management practices."
        ],
        "prevention": [
            "Maintain proper plant spacing.",
            "Avoid prolonged leaf wetness.",
            "Remove infected plant debris."
        ]
    },

    "Pepper__bell___healthy": {
        "name": "Pepper Bell Healthy",
        "type": "Healthy",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal plant care."
        ],
        "prevention": [
            "Maintain adequate watering.",
            "Provide sufficient sunlight.",
            "Monitor the plant regularly."
        ]
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected leaves.",
            "Improve air circulation.",
            "Avoid overhead watering.",
            "Use appropriate fungicide according to local recommendations and product label."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Maintain adequate plant spacing.",
            "Avoid prolonged leaf wetness."
        ]
    },

    "Potato___healthy": {
        "name": "Potato Healthy",
        "type": "Healthy",
        "treatment": [
            "No disease treatment is required.",
            "Continue regular crop monitoring."
        ],
        "prevention": [
            "Maintain proper irrigation.",
            "Provide adequate sunlight.",
            "Monitor plants regularly."
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected plant material.",
            "Avoid overhead irrigation.",
            "Improve air circulation.",
            "Use an appropriate fungicide according to local recommendations and product label."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Avoid prolonged leaf wetness.",
            "Monitor nearby plants for symptoms."
        ]
    },

    "Tomato__Target_Spot": {
        "name": "Tomato Target Spot",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected leaves.",
            "Improve air circulation.",
            "Avoid overhead watering.",
            "Use appropriate fungicide according to local recommendations and product label."
        ],
        "prevention": [
            "Maintain proper plant spacing.",
            "Remove infected plant debris.",
            "Avoid prolonged leaf wetness."
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "type": "Viral disease",
        "treatment": [
            "Remove severely infected plants or plant parts.",
            "Avoid handling healthy plants after infected plants.",
            "Disinfect gardening tools regularly."
        ],
        "prevention": [
            "Use clean planting material.",
            "Control weeds around the crop.",
            "Maintain good field hygiene."
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "type": "Viral disease",
        "treatment": [
            "Remove severely infected plants.",
            "Monitor and manage whitefly populations.",
            "Maintain good crop hygiene."
        ],
        "prevention": [
            "Monitor plants regularly.",
            "Control insect vectors such as whiteflies.",
            "Remove infected plant debris."
        ]
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "type": "Bacterial disease",
        "treatment": [
            "Remove severely affected leaves.",
            "Avoid overhead watering.",
            "Improve air circulation.",
            "Follow locally recommended bacterial disease management practices."
        ],
        "prevention": [
            "Maintain adequate plant spacing.",
            "Avoid prolonged leaf wetness.",
            "Remove infected plant debris."
        ]
    },

    "Tomato_Early_blight": {
        "name": "Tomato Early Blight",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected leaves.",
            "Avoid overhead watering.",
            "Improve air circulation.",
            "Use appropriate fungicide according to local recommendations and product label."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Maintain adequate plant spacing.",
            "Avoid prolonged leaf wetness."
        ]
    },

    "Tomato_healthy": {
        "name": "Tomato Healthy",
        "type": "Healthy",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal plant care."
        ],
        "prevention": [
            "Maintain adequate watering.",
            "Provide sufficient sunlight.",
            "Monitor the plant regularly."
        ]
    },

    "Tomato_Late_blight": {
        "name": "Tomato Late Blight",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected plant material.",
            "Avoid overhead watering.",
            "Improve air circulation.",
            "Use appropriate fungicide according to local recommendations and product label."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Avoid prolonged leaf wetness.",
            "Monitor nearby plants for symptoms."
        ]
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected leaves.",
            "Improve ventilation around plants.",
            "Avoid excessive humidity and leaf wetness."
        ],
        "prevention": [
            "Maintain good air circulation.",
            "Avoid overcrowding.",
            "Remove infected plant debris."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "type": "Fungal disease",
        "treatment": [
            "Remove severely affected lower leaves.",
            "Avoid overhead watering.",
            "Improve air circulation around plants.",
            "Use an appropriate fungicide according to local recommendations and the product label."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Maintain adequate plant spacing.",
            "Avoid prolonged leaf wetness."
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato Two-Spotted Spider Mite",
        "type": "Pest",
        "treatment": [
            "Inspect the underside of leaves.",
            "Remove heavily infested leaves.",
            "Use appropriate pest management according to local recommendations."
        ],
        "prevention": [
            "Monitor plants regularly.",
            "Maintain adequate plant health.",
            "Control infestations early."
        ]
    }
}


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error("Could not load the trained model.")
    st.code(str(e))
    st.stop()


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")
    resized = image.resize(IMG_SIZE)

    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][index])

    disease = class_names[index]

    return disease, confidence


# ------------------------------------------------------------
# Severity estimation
# ------------------------------------------------------------

def estimate_severity(confidence):

    if confidence >= 0.90:
        return "Moderate"
    elif confidence >= 0.70:
        return "Moderate"
    else:
        return "Low"


# ------------------------------------------------------------
# User Interface
# ------------------------------------------------------------

st.title("🌿 PlantCare AI")

st.subheader("Deep Learning Framework for Automated Plant Disease Detection")

st.write(
    "Upload a plant leaf image to detect the disease using "
    "the trained EfficientNet model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Leaf")
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("Prediction")

        if st.button("🔍 Analyze Leaf", use_container_width=True):

            with st.spinner("Analyzing leaf image..."):

                disease, confidence = predict_image(image)

                severity = estimate_severity(confidence)

                info = DISEASE_INFO.get(
                    disease,
                    {
                        "name": disease,
                        "type": "Unknown",
                        "treatment": [
                            "Consult a local agricultural expert."
                        ],
                        "prevention": [
                            "Monitor the plant regularly."
                        ]
                    }
                )

            st.success("Analysis completed successfully!")

            st.metric(
                "Detected Disease",
                info["name"]
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.metric(
                "Estimated Severity",
                severity
            )

            st.write(
                "**Disease Type:**",
                info["type"]
            )

            st.divider()

            st.subheader("🩺 Recommended Actions")

            for item in info["treatment"]:
                st.write("•", item)

            st.subheader("🌱 Prevention / Future Care")

            for item in info["prevention"]:
                st.write("•", item)

            st.info(
                "These are general AI-generated agricultural recommendations. "
                "Always follow the product label and local agricultural guidance "
                "before applying any treatment."
            )

else:

    st.info(
        "Please upload a plant leaf image to begin the analysis."
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "PlantCare AI | EfficientNet Transfer Learning | "
    "15 Plant Disease Classes"
)