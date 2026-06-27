from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
from pathlib import Path

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False

BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "brain_tumor_classifier.h5"
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma":     {"color": "#ff4d6d", "description": "Gliomas originate in the glial cells of the brain. They are the most common primary brain tumors and vary widely in severity."},
    "Meningioma": {"color": "#ff9f43", "description": "Meningiomas arise from the meninges surrounding the brain and spinal cord. Usually slow-growing and often benign."},
    "No Tumor":   {"color": "#22d3a5", "description": "No tumor detected in this MRI scan."},
    "Pituitary":  {"color": "#a29bfe", "description": "Pituitary tumors form in the pituitary gland at the base of the brain. Most are benign and treatable."}
}

MODEL_METRICS = {
    "test_accuracy": 87.00,
    "precision":     87.30,
    "recall":        87.00,
    "f1_score":      86.60,
    "model_type":    "EfficientNetB0",
    "input_size":    "224x224",
    "framework":     "TensorFlow 2.19",
    "num_classes":   4,
    "train_samples": 5712,
    "test_samples":  1600,
    "classes":       CLASS_NAMES,
    "per_class": {
        "Glioma":     {"precision": 93, "recall": 72, "f1": 81},
        "Meningioma": {"precision": 82, "recall": 78, "f1": 80},
        "No Tumor":   {"precision": 86, "recall": 99, "f1": 92},
        "Pituitary":  {"precision": 87, "recall": 99, "f1": 93}
    }
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
UPLOAD_FOLDER.mkdir(exist_ok=True)

model = None
model_load_error = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_brain_model():
    global model, model_load_error
    if not TENSORFLOW_AVAILABLE:
        model_load_error = "TensorFlow not available."
        return
    if not MODEL_PATH.exists():
        model_load_error = f"Model file not found: {MODEL_PATH.name}"
        return
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print(f"[INFO] Model loaded: {MODEL_PATH.name}")
    except Exception as e:
        model_load_error = str(e)
        model = None
        print(f"[ERROR] {model_load_error}")


load_brain_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_error": model_load_error})


@app.route("/metrics")
def metrics():
    return jsonify(MODEL_METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": f"Model not loaded: {model_load_error}"}), 503

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid format. Use PNG, JPG, or JPEG."}), 400

    try:
        processed_img = preprocess_image(file)
        raw_output    = model.predict(processed_img, verbose=0)[0]
        pred_index    = int(np.argmax(raw_output))
        predicted     = CLASS_NAMES[pred_index]
        confidence    = round(float(raw_output[pred_index]) * 100, 2)
        all_probs     = {CLASS_NAMES[i]: round(float(raw_output[i]) * 100, 2) for i in range(4)}

        return jsonify({
            "success":     True,
            "prediction":  predicted,
            "confidence":  confidence,
            "all_probs":   all_probs,
            "color":       CLASS_INFO[predicted]["color"],
            "description": CLASS_INFO[predicted]["description"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)