import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import gdown

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Model path
MODEL_PATH = BASE_DIR / "vgg_unfrozen.h5"

# YOUR GOOGLE DRIVE FILE ID
FILE_ID = "1oRq84RfUxUUCLrhrQbyJCY9A4Y1VpN0l"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ Download model using gdown (works for large files)
def download_model_if_needed():
    if MODEL_PATH.exists():
        print("Model already exists.")
        return

    print("Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(url, str(MODEL_PATH), quiet=False)

    print("Model downloaded successfully.")


# ✅ Build model architecture
def build_model():
    base_model = VGG19(include_top=False, weights=None, input_shape=(240, 240, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4608, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1152, activation="relu")(x)
    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.load_weights(str(MODEL_PATH))

    return model


def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"
    return "Unknown"


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((240, 240))

    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return int(np.argmax(prediction, axis=1)[0])


# 🔥 Load model at startup
download_model_if_needed()
model = build_model()
print("Model loaded successfully.")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        result_index = predict_image(str(file_path))
        result = get_class_name(result_index)
        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)