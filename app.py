from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------
# PlantCare AI — Expo-ready Streamlit application
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PlantCare AI | Leaf Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_efficientnet.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
EVALUATION_PATH = BASE_DIR / "reports" / "evaluation_metrics.json"
IMG_SIZE = (224, 224)
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

# The information below is general educational guidance, not a diagnosis or
# pesticide prescription. Confirm with a local agricultural professional.
DISEASE_INFO: dict[str, dict[str, Any]] = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper bacterial spot", "crop": "Pepper", "type": "Bacterial disease",
        "summary": "Bacterial spot can cause small, dark, water-soaked-looking lesions on leaves and may lead to leaf yellowing or drop.",
        "actions": ["Avoid handling plants while foliage is wet.", "Remove badly affected material when appropriate and dispose of it safely.", "Use clean tools and follow local extension guidance for management."],
        "prevention": ["Use healthy planting material.", "Avoid overhead irrigation where practical.", "Keep plants spaced for airflow and remove infected crop debris."]
    },
    "Pepper__bell___healthy": {
        "name": "Pepper — healthy class", "crop": "Pepper", "type": "Healthy class",
        "summary": "The model found the image most similar to the healthy pepper-leaf class in its training labels.",
        "actions": ["No disease treatment is indicated by this prediction alone.", "Continue normal crop care and inspect new growth."],
        "prevention": ["Water appropriately for the crop and conditions.", "Maintain airflow and balanced nutrition.", "Monitor leaves regularly for new symptoms."]
    },
    "Potato___Early_blight": {
        "name": "Potato early blight", "crop": "Potato", "type": "Fungal disease",
        "summary": "Early blight may produce brown leaf spots, sometimes with concentric rings. Other causes can look similar.",
        "actions": ["Remove severely affected leaves when appropriate.", "Reduce prolonged leaf wetness and improve airflow.", "Seek local crop guidance before selecting a fungicide; follow its label exactly."],
        "prevention": ["Remove infected plant debris.", "Rotate crops where feasible and maintain plant spacing.", "Avoid unnecessary overhead watering."]
    },
    "Potato___healthy": {
        "name": "Potato — healthy class", "crop": "Potato", "type": "Healthy class",
        "summary": "The image most closely matched the healthy potato-leaf class among the model's supported labels.",
        "actions": ["No disease treatment is indicated by this prediction alone.", "Continue routine crop monitoring."],
        "prevention": ["Use healthy seed tubers.", "Maintain appropriate irrigation and nutrition.", "Inspect plants regularly."]
    },
    "Potato___Late_blight": {
        "name": "Potato late blight", "crop": "Potato", "type": "Oomycete disease",
        "summary": "Late blight can progress quickly in suitable conditions. A photo model prediction should be confirmed promptly if symptoms are suspected.",
        "actions": ["Inspect nearby plants and seek local agricultural advice promptly.", "Avoid moving potentially infected plant material between areas.", "Use only locally recommended management and follow the product label."],
        "prevention": ["Use healthy planting material.", "Monitor plants and weather conditions closely.", "Remove infected material according to local guidance."]
    },
    "Tomato__Target_Spot": {
        "name": "Tomato target spot", "crop": "Tomato", "type": "Fungal disease",
        "summary": "Target spot can produce brown lesions on foliage; symptoms may overlap with other leaf diseases.",
        "actions": ["Remove badly affected leaves when appropriate.", "Reduce leaf wetness and improve airflow.", "Confirm the diagnosis locally before choosing a crop-protection product."],
        "prevention": ["Maintain suitable plant spacing.", "Remove infected debris.", "Avoid prolonged leaf wetness."]
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato mosaic virus", "crop": "Tomato", "type": "Viral disease",
        "summary": "Mosaic virus symptoms may include mottled leaf colour and distorted growth. Visual symptoms alone are not definitive.",
        "actions": ["Avoid handling healthy plants immediately after suspect plants.", "Clean tools and hands using suitable hygiene practices.", "Seek local guidance on removing and disposing of confirmed infected plants."],
        "prevention": ["Use healthy seed and planting material.", "Clean tools between plants.", "Remove volunteer plants and weeds that may maintain infection."]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato yellow leaf curl virus", "crop": "Tomato", "type": "Viral disease",
        "summary": "This virus may cause upward leaf curling, yellowing and stunted growth; whiteflies can transmit it.",
        "actions": ["Inspect plants for whiteflies and ask a local expert about integrated management.", "Confirm suspected viral symptoms before removing plants.", "There is no general photo-based cure recommendation."],
        "prevention": ["Use healthy planting material.", "Monitor whitefly populations.", "Follow locally recommended vector-management practices."]
    },
    "Tomato_Bacterial_spot": {
        "name": "Tomato bacterial spot", "crop": "Tomato", "type": "Bacterial disease",
        "summary": "Bacterial spot may cause small dark lesions on leaves and fruit. Similar symptoms can have other causes.",
        "actions": ["Avoid working with wet foliage.", "Remove badly affected material where appropriate.", "Use locally recommended management advice rather than selecting chemicals from an AI prediction alone."],
        "prevention": ["Use disease-free planting material.", "Clean tools and equipment.", "Avoid prolonged leaf wetness and remove infected debris."]
    },
    "Tomato_Early_blight": {
        "name": "Tomato early blight", "crop": "Tomato", "type": "Fungal disease",
        "summary": "Early blight often produces brown spots that may develop concentric rings, commonly on older leaves.",
        "actions": ["Remove severely affected leaves when appropriate.", "Improve airflow and reduce leaf wetness.", "Confirm the problem locally before using a fungicide."],
        "prevention": ["Remove infected plant debris.", "Maintain suitable plant spacing.", "Avoid unnecessary overhead irrigation."]
    },
    "Tomato_healthy": {
        "name": "Tomato — healthy class", "crop": "Tomato", "type": "Healthy class",
        "summary": "The image most closely matched the healthy tomato-leaf class among the model's supported labels.",
        "actions": ["No disease treatment is indicated by this prediction alone.", "Continue normal care and monitor new growth."],
        "prevention": ["Maintain appropriate watering and nutrition.", "Provide adequate airflow.", "Inspect plants regularly."]
    },
    "Tomato_Late_blight": {
        "name": "Tomato late blight", "crop": "Tomato", "type": "Oomycete disease",
        "summary": "Late blight can spread rapidly in favourable conditions. Confirm suspected cases promptly with local agricultural support.",
        "actions": ["Inspect nearby plants and seek local agricultural guidance promptly.", "Avoid moving suspect plant material around the crop.", "Use only locally recommended management and follow product labels."],
        "prevention": ["Monitor plants closely.", "Reduce prolonged leaf wetness where feasible.", "Handle and dispose of infected material according to local advice."]
    },
    "Tomato_Leaf_Mold": {
        "name": "Tomato leaf mold", "crop": "Tomato", "type": "Fungal disease",
        "summary": "Leaf mold is often associated with humid conditions and may cause yellow areas on the upper leaf surface with mold growth underneath.",
        "actions": ["Improve ventilation around plants.", "Remove affected leaves when appropriate.", "Seek local guidance if the disease is spreading."],
        "prevention": ["Avoid overcrowding.", "Manage humidity and leaf wetness where possible.", "Remove infected debris."]
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato Septoria leaf spot", "crop": "Tomato", "type": "Fungal disease",
        "summary": "Septoria leaf spot can cause small circular spots, often with pale centres and dark margins, especially on older leaves.",
        "actions": ["Remove severely affected lower leaves when appropriate.", "Avoid overhead watering and improve airflow.", "Confirm the diagnosis locally before selecting any fungicide."],
        "prevention": ["Remove infected debris.", "Maintain adequate plant spacing.", "Avoid prolonged leaf wetness."]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato two-spotted spider mite", "crop": "Tomato", "type": "Pest",
        "summary": "Spider mites may cause fine stippling or bronzing; check the undersides of leaves for mites or webbing.",
        "actions": ["Inspect the undersides of several leaves.", "Ask a local expert about suitable integrated pest management.", "Avoid applying pesticides without confirming the pest and checking the label."],
        "prevention": ["Monitor plants regularly.", "Reduce plant stress where practical.", "Protect beneficial insects and follow local guidance."]
    },
}

# Friendly names also support minor spelling/underscore differences in class files.
CLASS_ALIASES = {
    "Pepper___Bacterial_spot": "Pepper__bell___Bacterial_spot",
    "Pepper___healthy": "Pepper__bell___healthy",
    "Tomato_Tomato_mosaic_virus": "Tomato__Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Tomato_YellowLeaf__Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
}

st.markdown("""
<style>
:root { --pc-green:#1f8a5b; --pc-deep:#12372a; --pc-mint:#e9f7ef; }
.block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f7fbf8 0%, #ffffff 42%, #f7faf8 100%); }
[data-testid="stSidebar"] { background: #102f25; }
[data-testid="stSidebar"] * { color: #f3fbf6 !important; }
.pc-hero { background: linear-gradient(120deg,#12372a 0%,#1e6b4a 58%,#62a87a 100%); color:white; padding: 2rem 2.1rem; border-radius: 24px; margin-bottom: 1.2rem; box-shadow: 0 12px 32px rgba(18,55,42,.16); }
.pc-kicker { text-transform: uppercase; letter-spacing: .16em; font-size:.75rem; font-weight:700; opacity:.82; }
.pc-hero h1 { font-size: clamp(2rem,4vw,3.2rem); line-height:1.08; margin:.55rem 0 .7rem 0; color:white; }
.pc-hero p { font-size:1.03rem; max-width:800px; opacity:.93; margin-bottom:.4rem; }
.pc-pill { display:inline-block; background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.28); border-radius:999px; padding:.32rem .7rem; margin:.2rem .25rem .1rem 0; font-size:.78rem; }
.pc-section { font-size:1.28rem; font-weight:750; color:#153e2e; margin:.3rem 0 .7rem; }
.pc-card { background:white; border:1px solid #e2eee6; border-radius:18px; padding:1rem 1.1rem; box-shadow:0 4px 18px rgba(24,67,45,.045); height:100%; }
.pc-label { color:#62766a; font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.pc-value { color:#173e2d; font-size:1.45rem; font-weight:800; margin-top:.2rem; overflow-wrap:anywhere; }
.pc-small { color:#5f7166; font-size:.86rem; }
.pc-note { background:#fff8e8; border:1px solid #f0dfb7; border-radius:12px; padding:.85rem 1rem; color:#684f18; font-size:.9rem; }
.pc-footer { text-align:center; color:#6b7c72; font-size:.8rem; padding-top:1.2rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e2eee6; padding: .9rem 1rem; border-radius:16px; }
div[data-testid="stFileUploader"] { background:white; border:1px dashed #9fcab0; border-radius:16px; padding:.5rem; }
.stButton button[kind="primary"] { background:#1f8a5b; border-color:#1f8a5b; border-radius:12px; font-weight:700; min-height:3rem; }
/* PlantCare AI — warm botanical expo theme */
:root {
  --pc-forest: #0f3b2d;
  --pc-green: #16834f;
  --pc-leaf: #5aa86f;
  --pc-cream: #f7f8ed;
  --pc-card: #fffef8;
  --pc-ink: #173b31;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 8% 8%, rgba(190,222,169,.42) 0%, transparent 28%),
    radial-gradient(ellipse at 94% 84%, rgba(179,218,161,.38) 0%, transparent 26%),
    linear-gradient(135deg, #f5f7e9 0%, #fbfbf4 48%, #edf5e8 100%);
  color: var(--pc-ink);
}
[data-testid="stHeader"] {
  background: rgba(247,248,237,.88);
}
[data-testid="stSidebar"] {
  background: linear-gradient(165deg, #0c3528 0%, #14543b 58%, #0d3a2b 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #f3fbf4 !important; }
.block-container {
  max-width: 1370px;
  padding-top: 1.6rem;
  padding-bottom: 2.8rem;
}
.pc-hero {
  background: linear-gradient(115deg, #123d2e 0%, #18734b 56%, #75b987 100%);
  color: white;
  padding: 2.15rem 2.25rem;
  border-radius: 26px;
  margin-bottom: 1.5rem;
  box-shadow: 0 14px 34px rgba(25,78,49,.15);
  border: 1px solid rgba(255,255,255,.25);
}
.pc-hero h1 { color: white !important; }
.pc-section {
  color: #124b36;
  font-size: 1.24rem;
  font-weight: 800;
  margin: .45rem 0 .8rem;
}
.pc-card {
  background: rgba(255,255,250,.94);
  border: 1px solid #dcebd8;
  border-radius: 20px;
  padding: 1.15rem 1.25rem;
  box-shadow: 0 7px 24px rgba(34,82,49,.07);
  height: 100%;
}
.pc-label { color: #64806c; }
.pc-value { color: #174a35; }
.pc-small { color: #5b7062; }
.pc-note {
  background: #fff5df;
  border: 1px solid #f4d99f;
  border-radius: 15px;
  padding: .95rem 1.1rem;
  color: #684a18;
  font-size: .91rem;
  box-shadow: 0 4px 14px rgba(120,89,31,.04);
}
.pc-footer { color: #557364; }
div[data-testid="stMetric"] {
  background: rgba(255,255,250,.96);
  border: 1px solid #dcebd8;
  border-radius: 17px;
  padding: 1rem 1.1rem;
  box-shadow: 0 5px 18px rgba(34,82,49,.05);
}
div[data-testid="stFileUploader"] {
  background: rgba(255,255,250,.92);
  border: 1.5px dashed #9bc9a6;
  border-radius: 17px;
  padding: .65rem;
}
.stButton button[kind="primary"] {
  background: linear-gradient(100deg,#16834f,#279760);
  border: 0;
  color: white;
  border-radius: 13px;
  font-weight: 750;
  min-height: 3.15rem;
  box-shadow: 0 5px 13px rgba(22,131,79,.18);
}
.stButton button[kind="primary"]:hover {
  background: linear-gradient(100deg,#116b40,#1f824f);
  border: 0;
  color: white;
}
div[data-testid="stDownloadButton"] button {
  background: #16834f;
  color: white;
  border: 0;
  border-radius: 12px;
  font-weight: 700;
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  background: #fffef8;
  border: 1px solid #dcebd8;
  border-radius: 14px;
  overflow: hidden;
}
div[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stExpander"] {
  background: rgba(255,255,250,.75);
  border: 1px solid #dcebd8;
  border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained EfficientNet model…")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Copy your trained .keras model into the models folder."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class list not found at {CLASS_PATH}. Copy class_names.json into the models folder."
        )
    with CLASS_PATH.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = [classes[str(i)] for i in range(len(classes))]
    if not isinstance(classes, list) or not classes:
        raise ValueError("class_names.json must contain a non-empty list of class names.")
    return [str(c) for c in classes]


def normalise_class_key(class_name: str) -> str:
    return CLASS_ALIASES.get(class_name, class_name)


def get_disease_info(class_name: str) -> dict[str, Any]:
    key = normalise_class_key(class_name)
    if key in DISEASE_INFO:
        return DISEASE_INFO[key]
    return {
        "name": class_name.replace("_", " ").replace("__", " ").strip(),
        "crop": "Not mapped",
        "type": "Class label",
        "summary": "No curated description is available for this class label.",
        "actions": ["Treat this result as a screening hint and seek confirmation from a local agricultural expert."],
        "prevention": ["Monitor the plant and keep a record of changing symptoms."]
    }


def predict_image(image: Image.Image, model, class_names: list[str]):
    # The EfficientNetB0 model in this project uses its built-in input rescaling.
    # Pass RGB pixel values in the same 0–255 range used during training.
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    scores = np.asarray(model.predict(batch, verbose=0))[0].astype(float)
    if len(scores) != len(class_names):
        raise ValueError(
            f"Model returns {len(scores)} scores, but class_names.json contains {len(class_names)} classes."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction scores.")
    order = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    return {
        "class_name": class_names[best_idx],
        "confidence": float(scores[best_idx]),
        "top_predictions": [
            {"class_name": class_names[int(i)], "score": float(scores[int(i)])}
            for i in order[: min(3, len(order))]
        ],
        "image": rgb,
        "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()[:16],
    }


def generate_gradcam(image: Image.Image, model, class_index: int):
    """Generate Grad-CAM for the project's nested EfficientNet + GAP + Dense head."""
    base_model = next(
        (layer for layer in model.layers
         if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()),
        None,
    )
    if base_model is None:
        raise RuntimeError("Could not locate the nested EfficientNet feature extractor.")

    target_layer = next(
        (layer for layer in reversed(base_model.layers)
         if getattr(getattr(layer, "output", None), "shape", None) is not None
         and len(layer.output.shape) == 4),
        None,
    )
    gap_layer = next((layer for layer in model.layers if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)), None)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    output_layer = dense_layers[-1] if dense_layers else None
    dropout_layer = next((layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)), None)
    if target_layer is None or gap_layer is None or output_layer is None:
        raise RuntimeError("The model layers do not match the expected EfficientNet/GAP/Dense architecture.")

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
    )
    img = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    tensor = tf.convert_to_tensor(np.expand_dims(np.asarray(img, dtype=np.float32), axis=0))
    with tf.GradientTape() as tape:
        feature_maps, base_output = grad_model(tensor, training=False)
        pooled = gap_layer(base_output)
        if dropout_layer is not None:
            pooled = dropout_layer(pooled, training=False)
        logits = output_layer(pooled)
        class_score = logits[:, class_index]
    grads = tape.gradient(class_score, feature_maps)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients were unavailable for this model.")
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray):
    import matplotlib.cm as cm
    original = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(IMG_SIZE, Image.Resampling.BILINEAR)
    heat_rgb = cm.get_cmap("turbo")(np.asarray(heat) / 255.0)[..., :3]
    overlay = np.clip(0.58 * original + 0.42 * heat_rgb, 0, 1)
    return Image.fromarray(np.uint8(overlay * 255))


def build_report(result: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "application": "PlantCare AI",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "EfficientNetB0 transfer learning",
        "supported_classes": 15,
        "prediction": info["name"],
        "class_label": result["class_name"],
        "model_score": round(result["confidence"], 6),
        "model_score_percent": round(result["confidence"] * 100, 2),
        "top_predictions": [
            {"class_label": x["class_name"], "score_percent": round(x["score"] * 100, 2)}
            for x in result["top_predictions"]
        ],
        "image_fingerprint_sha256_prefix": result["image_sha256"],
        "limitations": [
            "The model was trained on PlantVillage leaf images; field photos may differ from the training domain.",
            "The displayed score is a model output, not a calibrated probability of correctness.",
            "This screening output is not a confirmed diagnosis or a pesticide prescription.",
            "Grad-CAM highlights model-relevant regions; it does not prove disease causation.",
        ],
    }


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    st.caption("Leaf intelligence • Research prototype")
    st.markdown("---")
    st.markdown("**Supported crops**")
    st.markdown("🍅 Tomato  \n🥔 Potato  \n🫑 Pepper")
    st.markdown("**Model**")
    st.markdown("EfficientNetB0  \n224 × 224 RGB input  \n15 training classes")
    st.markdown("---")
    st.markdown("**How to get a useful result**")
    st.markdown("1. Use a clear, in-focus leaf photo.  \n2. Keep one leaf prominent.  \n3. Use even, natural light.  \n4. Avoid screenshots, filters and busy backgrounds.")
    st.caption("Prototype only • Not a substitute for an agricultural diagnosis")

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown("""
<section class="pc-hero">
  <div class="pc-kicker">AI-assisted crop health screening</div>
  <h1>See the signal in every leaf.</h1>
  <p>Upload a tomato, potato or pepper leaf image to explore the model's predicted class,
  its top alternatives, a Grad-CAM visual explanation, and practical next steps.</p>
  <span class="pc-pill">EfficientNetB0</span>
  <span class="pc-pill">Transfer learning</span>
  <span class="pc-pill">Explainable AI</span>
  <span class="pc-pill">15 classes</span>
</section>
""", unsafe_allow_html=True)

try:
    model = load_model()
    class_names = load_classes()
except Exception as exc:
    st.error("The application could not load its model assets.")
    st.warning("Place both files in the project's `models` folder, then restart Streamlit.")
    st.code(f"Expected model: {MODEL_PATH}\nExpected classes: {CLASS_PATH}")
    st.exception(exc)
    st.stop()

if len(class_names) != 15:
    st.warning(f"The loaded class list contains {len(class_names)} classes. The sidebar's 15-class description should be updated if this is intentional.")

left, right = st.columns([1.02, 0.98], gap="large")
with left:
    st.markdown('<div class="pc-section">01 · Upload a leaf</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=SUPPORTED_TYPES,
        help="Supported formats: JPG, JPEG, PNG, WEBP and BMP.",
    )
    st.caption("Tip: photograph the leaf close-up without flash glare. The model has not been validated on every field condition.")
    if uploaded_file is not None:
        try:
            source_image = Image.open(uploaded_file)
            source_image = ImageOps.exif_transpose(source_image).convert("RGB")
            st.image(source_image, caption="Image ready for analysis", use_container_width=True)
            st.caption(f"Image size: {source_image.width} × {source_image.height} pixels")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Could not read this image. Try another JPG or PNG file. Details: {exc}")
            source_image = None
    else:
        source_image = None
        st.markdown("""
        <div class="pc-card">
          <div class="pc-label">Ready when you are</div>
          <div class="pc-value">Upload a leaf photo</div>
          <p class="pc-small">A clear single-leaf image generally makes results easier to inspect.</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="pc-section">02 · Run the screening</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pc-card">
      <div class="pc-label">Analysis pipeline</div>
      <div class="pc-value">Image → EfficientNet → Explanation</div>
      <p class="pc-small">The image is resized to 224 × 224 RGB pixels. The model scores the supported training classes; optional Grad-CAM shows regions that influenced the selected class.</p>
    </div>
    """, unsafe_allow_html=True)
    analyze_clicked = st.button(
        "🌱 Analyze leaf image",
        type="primary",
        use_container_width=True,
        disabled=source_image is None,
    )
    show_explanation = st.toggle("Generate Grad-CAM visual explanation", value=True)
    st.caption("Grad-CAM adds processing time. Turn it off for a faster prediction.")

if analyze_clicked and source_image is not None:
    try:
        with st.spinner("Analysing the leaf image…"):
            result = predict_image(source_image, model, class_names)
            info = get_disease_info(result["class_name"])
            report = build_report(result, info)
            gradcam_image = None
            gradcam_note = None
            if show_explanation:
                try:
                    heatmap = generate_gradcam(source_image, model, class_names.index(result["class_name"]))
                    gradcam_image = overlay_gradcam(source_image, heatmap)
                except Exception as exc:
                    gradcam_note = str(exc)
            st.session_state["plantcare_result"] = {
                "result": result,
                "info": info,
                "report": report,
                "gradcam_image": gradcam_image,
                "gradcam_note": gradcam_note,
                "uploaded_name": uploaded_file.name if uploaded_file else "uploaded-image",
            }
    except Exception as exc:
        st.error("Analysis failed. Check that the model and class-name files belong to the same training run.")
        st.exception(exc)

saved = st.session_state.get("plantcare_result")
if saved:
    result = saved["result"]
    info = saved["info"]
    report = saved["report"]
    confidence_pct = result["confidence"] * 100

    st.markdown("---")
    st.markdown('<div class="pc-section">03 · Screening result</div>', unsafe_allow_html=True)
    top_cols = st.columns([1.25, 0.8, 0.8], gap="medium")
    with top_cols[0]:
        st.markdown(f"""
        <div class="pc-card">
          <div class="pc-label">Top predicted class</div>
          <div class="pc-value">{info["name"]}</div>
          <p class="pc-small">{info["crop"]} · {info["type"]}</p>
        </div>
        """, unsafe_allow_html=True)
    with top_cols[1]:
        st.metric("Model score", f"{confidence_pct:.2f}%")
        st.caption("Score is not a guarantee of correctness.")
    with top_cols[2]:
        st.metric("Supported classes", len(class_names))
        st.caption("From this model's label set.")

    st.progress(min(max(float(result["confidence"]), 0.0), 1.0), text=f"Top class score: {confidence_pct:.2f}%")
    st.markdown(f"**What this class means:** {info['summary']}")
    st.markdown("""
    <div class="pc-note"><strong>Important:</strong> this is an AI screening result, not a confirmed diagnosis. The model was trained on PlantVillage-style leaf images; real field images, lighting, crop variety and other stresses can change performance. Confirm uncertain or high-impact cases with a local agricultural expert.</div>
    """, unsafe_allow_html=True)

    details_col, explanation_col = st.columns([1, 1], gap="large")
    with details_col:
        st.markdown('<div class="pc-section">Top model alternatives</div>', unsafe_allow_html=True)
        top_rows = []
        for item in result["top_predictions"]:
            friendly = get_disease_info(item["class_name"])["name"]
            top_rows.append({"Class": friendly, "Model score": f"{item['score'] * 100:.2f}%"})
        st.table(top_rows)
        st.caption("These are the model's highest-scoring classes, not independent confirmations.")
    with explanation_col:
        st.markdown('<div class="pc-section">Visual explanation · Grad-CAM</div>', unsafe_allow_html=True)
        if saved["gradcam_image"] is not None:
            st.image(saved["gradcam_image"], caption="Highlighted regions influenced the selected class. This is not a disease boundary or severity measurement.", use_container_width=True)
        elif saved["gradcam_note"]:
            st.info("Grad-CAM could not be generated for this model architecture. The class prediction is still available.")
            with st.expander("Technical details"):
                st.code(saved["gradcam_note"])
        else:
            st.info("Grad-CAM was switched off for this analysis.")

    action_col, prevention_col = st.columns(2, gap="large")
    with action_col:
        st.markdown('<div class="pc-section">Practical next steps</div>', unsafe_allow_html=True)
        for item in info["actions"]:
            st.markdown(f"- {item}")
    with prevention_col:
        st.markdown('<div class="pc-section">Prevention and monitoring</div>', unsafe_allow_html=True)
        for item in info["prevention"]:
            st.markdown(f"- {item}")

    st.markdown("""
    <div class="pc-note"><strong>Safe-use note:</strong> PlantCare AI does not prescribe a pesticide or dosage. Before applying any crop-protection product, confirm the problem, check that the product is approved for the crop and location, and follow the label and local agricultural guidance.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pc-section">Download your analysis</div>', unsafe_allow_html=True)
    json_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "⬇️ Download screening report (JSON)",
        data=json_bytes,
        file_name=f"plantcare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=False,
    )
    with st.expander("View report details"):
        st.json(report)

st.markdown("---")
with st.expander("Model evaluation results and limitations"):
    if EVALUATION_PATH.exists():
        try:
            evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
            st.markdown("**Saved held-out test-set results**")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Test accuracy", f"{evaluation['accuracy_percent']:.2f}%")
            metric_cols[1].metric("Weighted precision", f"{evaluation['weighted_precision_percent']:.2f}%")
            metric_cols[2].metric("Weighted recall", f"{evaluation['weighted_recall_percent']:.2f}%")
            metric_cols[3].metric("Weighted F1", f"{evaluation['weighted_f1_percent']:.2f}%")
            st.caption(f"Evaluated {evaluation['test_images']} held-out images · {evaluation['class_count']} classes · Saved {evaluation.get('generated_at', 'date not recorded')}")
            cm_path = BASE_DIR / "reports" / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix for the saved held-out test evaluation", use_container_width=True)
            st.warning("These metrics describe the project's held-out dataset split, not guaranteed accuracy on real farm images.")
        except (KeyError, ValueError, OSError, TypeError) as exc:
            st.warning(f"Evaluation report exists but could not be read: {exc}")
    else:
        st.info("No saved evaluation report found yet. Run `python evaluate_model.py` to generate genuine metrics for this model and dataset.")
    st.markdown("""
    **Model:** EfficientNetB0 with transfer learning, using 224 × 224 RGB images.

    **Training dataset:** the project scripts target a 15-class PlantVillage folder structure. Use the dataset count printed by `check_dataset.py` and the actual training logs in your report.

    **Evaluation:** report the held-out test-set metrics produced by `evaluate_model.py`. Do not describe those results as real-world field accuracy unless an independent field-image evaluation has been conducted.

    **Limitations:** model scores may be overconfident; the supported labels are limited to the classes listed in `models/class_names.json`; Grad-CAM is an explanation aid rather than causal evidence; severity is intentionally not inferred from prediction confidence or Grad-CAM area because that would require separate validation.
    """)

st.markdown("""
<div class="pc-footer">
  <strong>PlantCare AI</strong> · EfficientNetB0 · Transfer learning · Explainable AI<br/>
  Research prototype for educational use. Verify findings before making crop-management decisions.
</div>
""", unsafe_allow_html=True)
