"""
DeepFER - Facial Emotion Recognition
Streamlit frontend for the trained CNN model.

Run locally with:
    pip install streamlit tensorflow pillow numpy opencv-python-headless
    streamlit run app.py

Before running, place your trained model file (downloaded from Kaggle)
in the same folder as this script, named either:
    deepfer_cnn_model.keras   (final saved model)
    best_fer_model.keras      (best checkpoint saved during training)
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepFER — Facial Emotion Recognition",
    page_icon="🙂",
    layout="centered",
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# Colors per class for the confidence bar chart
COLORS = {
    "angry": "#E4572E",
    "disgust": "#8E6C88",
    "fear": "#5C4B99",
    "happy": "#F3A712",
    "neutral": "#7A8B99",
    "sad": "#3A6EA5",
    "surprise": "#38B000",
}

MODEL_PATHS = ["deepfer_cnn_model.keras", "best_fer_model.keras"]
IMG_SIZE = (48, 48)


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model = keras.models.load_model(path)
            return model, path
    return None, None


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to the (1, 48, 48, 1) normalized array the model expects."""
    img = pil_img.convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return arr


def try_face_crop(pil_img: Image.Image) -> Image.Image:
    """Best-effort face crop using OpenCV's Haar cascade, if available.
    Falls back to the original image if no face is detected or cv2/haar file is missing."""
    try:
        import cv2
        img_cv = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest detected face
            face_img = gray[y:y + h, x:x + w]
            return Image.fromarray(face_img)
    except Exception:
        pass
    return pil_img


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🙂 DeepFER — Facial Emotion Recognition")
st.caption(
    "A CNN trained from scratch on FER2013 (48×48 grayscale faces, 7 emotion classes). "
    "Upload a face photo or take one with your webcam."
)

model, loaded_path = load_model()

if model is None:
    st.error(
        "No trained model found. Place **deepfer_cnn_model.keras** or "
        "**best_fer_model.keras** in the same folder as this app, then refresh."
    )
    st.stop()
else:
    st.success(f"Model loaded: `{loaded_path}`")

with st.expander("ℹ️ About this model — read before trusting the result"):
    st.markdown(
        """
        - Trained on **FER2013**, a well-known but difficult benchmark dataset.
        - Test accuracy achieved: **~58%**, with known benchmark ceilings around **70–75%**
          and human label agreement itself only around **~65%** — emotion labels in this
          dataset are inherently ambiguous.
        - **Disgust** is the weakest class (only ~436 training images vs. ~7,200 for "happy"),
          so predictions of disgust should be treated with lower confidence.
        - **Fear** is also commonly confused with **surprise** and **sad** due to visual similarity.
        - This tool is a portfolio/demo project, not a validated clinical or commercial product.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Input: upload or webcam
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Use Webcam"])

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a face photo (jpg/png)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        input_image = Image.open(camera_file)

auto_crop = st.checkbox(
    "Try to auto-detect and crop the face before prediction (recommended)", value=True
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if input_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(input_image, use_container_width=True)

    display_image = try_face_crop(input_image) if auto_crop else input_image

    with col2:
        st.subheader("Model sees (48×48 grayscale)")
        preview = display_image.convert("L").resize(IMG_SIZE)
        st.image(preview, width=150)

    processed = preprocess_image(display_image)
    preds = model.predict(processed, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx])

    st.divider()
    st.markdown(
        f"## {EMOJI[top_class]} Predicted emotion: **{top_class.capitalize()}**  "
        f"`{top_conf * 100:.1f}% confidence`"
    )

    if top_class == "disgust":
        st.warning(
            "⚠️ 'Disgust' predictions are the least reliable from this model — it was "
            "trained on very little data for this class. Double-check with a different photo."
        )
    if top_conf < 0.4:
        st.info(
            "ℹ️ Low confidence prediction — the model is unsure. This can happen with "
            "ambiguous expressions, poor lighting, or a non-frontal face angle."
        )

    st.subheader("Confidence across all classes")
    sorted_pairs = sorted(zip(CLASS_NAMES, preds), key=lambda p: p[1], reverse=True)
    for cls, prob in sorted_pairs:
        st.markdown(f"**{EMOJI[cls]} {cls.capitalize()}** — {prob * 100:.1f}%")
        st.progress(float(prob))

else:
    st.info("Upload an image or take a photo above to get a prediction.")

st.divider()
st.caption(
    "DeepFER · Built with TensorFlow/Keras + Streamlit · "
    "Trained on FER2013 · Ahad Ahmad"
)
