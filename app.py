from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "model.h5"

# Class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load model once when app starts
model = None
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model_error = f"Model file not found: {MODEL_PATH}"
        print(model_error)
except Exception as e:
    model_error = f"Error loading model: {str(e)}"
    print(model_error)


def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Prediction request received.")

        if model is None:
            return jsonify({
                "success": False,
                "error": model_error if model_error else "Model is not loaded."
            }), 500

        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file part in request."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        allowed_extensions = {"png", "jpg", "jpeg"}
        file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

        if file_ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": "Invalid file format. Please upload PNG, JPG, or JPEG."
            }), 400

        processed_img = preprocess_image(file)

        prediction = model.predict(processed_img)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction) * 100)

        result = {
            "success": True,
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 2)
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)