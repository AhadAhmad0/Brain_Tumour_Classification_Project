import os
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "final_model.h5"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"
    return "Unknown"


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


print("Base directory:", BASE_DIR)
print("Files in directory:", os.listdir(BASE_DIR))
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

try:
    model = load_model(str(MODEL_PATH), compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", str(e))
    model = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server. Check Render logs."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PNG, JPG, and JPEG files are allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename

    try:
        file.save(file_path)

        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)

        class_index = int(np.argmax(prediction, axis=1)[0])
        result = get_class_name(class_index)
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print("Could not delete uploaded file:", str(e))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
