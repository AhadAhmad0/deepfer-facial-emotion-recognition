"""
DeepFER - Facial Emotion Recognition
Gradio frontend for the trained CNN model. Built for Hugging Face Spaces
(Gradio SDK) but also runs fine locally.

Run locally with:
    pip install gradio tensorflow pillow numpy opencv-python-headless
    python app.py

Before running, place your trained model file (downloaded from Kaggle)
in the same folder as this script, named either:
    deepfer_cnn_model.keras   (final saved model)
    best_fer_model.keras      (best checkpoint saved during training)
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

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

MODEL_PATHS = ["deepfer_cnn_model.keras", "best_fer_model.keras"]
IMG_SIZE = (48, 48)

ABOUT_TEXT = """
### About this model — read before trusting the result

- Trained from scratch on **FER2013**, a well-known but difficult benchmark dataset.
- Test accuracy achieved: **~58%**, with known benchmark ceilings around **70-75%**
  and human label agreement itself only around **~65%** — emotion labels in this
  dataset are inherently ambiguous.
- **Disgust** is the weakest class (only ~436 training images vs. ~7,200 for "happy"),
  so predictions of disgust should be treated with lower confidence.
- **Fear** is also commonly confused with **surprise** and **sad** due to visual similarity.
- This is a portfolio/demo project, not a validated clinical or commercial product.
"""


def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return keras.models.load_model(path), path
    return None, None


model, loaded_path = load_model()


def try_face_crop(pil_img: Image.Image) -> Image.Image:
    """Best-effort face crop using OpenCV's Haar cascade. Falls back to the
    original image if no face is detected or cv2 isn't available."""
    try:
        import cv2
        img_cv = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_img = gray[y:y + h, x:x + w]
            return Image.fromarray(face_img)
    except Exception:
        pass
    return pil_img


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)


def predict(input_image, auto_crop):
    if input_image is None:
        return None, "Upload an image or take a photo to get a prediction.", None

    if model is None:
        return None, "No trained model file found on the server.", None

    pil_img = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image

    display_image = try_face_crop(pil_img) if auto_crop else pil_img
    processed = preprocess_image(display_image)
    preds = model.predict(processed, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx])

    label_dict = {f"{EMOJI[c]} {c.capitalize()}": float(p) for c, p in zip(CLASS_NAMES, preds)}

    warning = ""
    if top_class == "disgust":
        warning += (
            "\n\n**'Disgust' predictions are the least reliable from this model** — "
            "it was trained on very little data for this class. Try a different photo to confirm."
        )
    if top_conf < 0.4:
        warning += (
            "\n\n**Low confidence prediction** — the expression may be ambiguous, "
            "lighting/angle may be poor, or the face wasn't detected cleanly."
        )

    model_view = display_image.convert("L").resize((150, 150))

    result_text = f"## {EMOJI[top_class]} Predicted: **{top_class.capitalize()}** ({top_conf * 100:.1f}% confidence){warning}"

    return label_dict, result_text, model_view


with gr.Blocks(title="DeepFER — Facial Emotion Recognition") as demo:
    gr.Markdown("# DeepFER — Facial Emotion Recognition")
    gr.Markdown(
        "A CNN trained from scratch on FER2013 (48x48 grayscale faces, 7 emotion classes). "
        "Upload a face photo or use your webcam."
    )

    if model is not None:
        gr.Markdown(f"Model loaded: `{loaded_path}`")
    else:
        gr.Markdown(
            "**No trained model found.** Place `deepfer_cnn_model.keras` or "
            "`best_fer_model.keras` in this Space's root folder."
        )

    with gr.Accordion("About this model — read before trusting the result", open=False):
        gr.Markdown(ABOUT_TEXT)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload or capture a face photo", sources=["upload", "webcam"])
            auto_crop_checkbox = gr.Checkbox(value=True, label="Auto-detect and crop the face before prediction (recommended)")
            predict_btn = gr.Button("Predict Emotion", variant="primary")

        with gr.Column():
            result_md = gr.Markdown()
            confidence_output = gr.Label(label="Confidence across all classes", num_top_classes=7)
            model_view_output = gr.Image(label="What the model actually sees (48x48 grayscale)", width=150)

    predict_btn.click(
        fn=predict,
        inputs=[image_input, auto_crop_checkbox],
        outputs=[confidence_output, result_md, model_view_output],
    )

    gr.Markdown("---")
    gr.Markdown("DeepFER · Built with TensorFlow/Keras + Gradio · Trained on FER2013 · Ahad Ahmad")


if __name__ == "__main__":
    demo.launch()
