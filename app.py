from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload
UPLOAD_FOLDER.mkdir(exist_ok=True)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": False,
        "mode": "demo"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file format"
            }), 400

        # Still preprocess to validate image
        _ = preprocess_image(file)

        # Demo prediction
        prediction = random.choice(CLASS_NAMES)
        confidence = round(random.uniform(85, 99), 2)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "note": "Demo mode prediction"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)