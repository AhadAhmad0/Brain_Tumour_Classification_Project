from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.h5"
UPLOAD_FOLDER = BASE_DIR / "uploads"

# App config
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max

UPLOAD_FOLDER.mkdir(exist_ok=True)

# Classes (IMPORTANT: must match training)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = None
model_error = None


# ✅ Load model safely
def load_brain_model():
    global model, model_error
    try:
        print("Checking files:", os.listdir(BASE_DIR))

        if not MODEL_PATH.exists():
            model_error = f"Model file not found: {MODEL_PATH.name}"
            print(model_error)
            return

        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

    except Exception as e:
        model_error = f"Error loading model: {str(e)}"
        print(model_error)


load_brain_model()


# ✅ Helper: check file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ Preprocess image
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ✅ Home route
@app.route("/")
def home():
    return render_template("index.html")


# ✅ Health check (optional but useful)
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "error": model_error
    })


# ✅ FIXED Predict Route (NO unreachable code)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check model
        if model is None:
            return jsonify({
                "success": False,
                "error": model_error if model_error else "Model not loaded"
            }), 500

        # Check file exists
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # Check filename
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Check extension
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file format"
            }), 400

        # Preprocess
        processed_img = preprocess_image(file)

        # Predict
        prediction = model.predict(processed_img, verbose=0)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "success": True,
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ✅ Render-compatible run
if __name__ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)