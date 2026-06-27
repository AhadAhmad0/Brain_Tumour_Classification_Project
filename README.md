# 🧠 Brain Tumor Classification

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?logo=flask)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?logo=docker)](https://docker.com)

A deep learning web application that classifies brain MRI scans into 4 tumor categories using **EfficientNetB0** with transfer learning. Deployed on Hugging Face Spaces via Docker.

---

## 🔴 Live Demo

👉 **[Try it here](https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification)**

Upload any brain MRI scan and get an instant prediction with confidence scores for all 4 classes.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **87.00%** |
| Precision (weighted) | **87.30%** |
| Recall (weighted) | **87.00%** |
| F1 Score (weighted) | **86.60%** |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 93% | 72% | 81% | 400 |
| Meningioma | 82% | 78% | 80% | 400 |
| No Tumor | 86% | 99% | 92% | 400 |
| Pituitary | 87% | 99% | 93% | 400 |

---

## 🗂️ Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by Masoud Nickparvar

| Split | Images |
|-------|--------|
| Training | 5,712 |
| Testing | 1,600 |
| **Total** | **7,023** |

4 classes: `Glioma` · `Meningioma` · `No Tumor` · `Pituitary`

---

## 🏗️ Architecture

```
EfficientNetB0 (ImageNet weights, frozen)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(256, relu)
    └── Dropout(0.4)
    └── Dense(128, relu)
    └── Dropout(0.3)
    └── Dense(4, softmax)
```

**Training strategy:**
- Phase 1 — Frozen base, 10 epochs, lr=1e-3
- Phase 2 — Fine-tune last 30 layers, 8 epochs, lr=1e-5
- EarlyStopping + ReduceLROnPlateau callbacks
- No manual rescaling (EfficientNetB0 handles normalization internally)

---

## 📈 Training Curves

<img width="2085" height="731" alt="training_curves" src="https://github.com/user-attachments/assets/f3999d00-d189-44be-ba1d-59b80b1a1c21" />


## 🔢 Confusion Matrix

<img width="1111" height="882" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b2fa6a9f-9bdf-4251-9df4-3ecc80132e87" />


---

## 🚀 Deployment

Deployed on **Hugging Face Spaces** using Docker.

```
app.py                        # Flask backend
templates/index.html          # Frontend UI
brain_tumor_classifier.h5     # Trained model (hosted on HF Spaces)
Dockerfile                    # Docker configuration
requirements.txt              # Python dependencies
```

> **Note:** The model file (`brain_tumor_classifier.h5`, ~32MB) is hosted on Hugging Face Spaces via Git LFS and is not included in this GitHub repository.

---

## ⚙️ Run Locally

```bash
git clone https://github.com/AhadAhmad0/Brain_Tumour_Classification_Project
cd Brain_Tumour_Classification_Project

pip install -r requirements.txt

# Download model from Hugging Face and place in project root
# https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification

python app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Model | EfficientNetB0 (Transfer Learning) |
| Framework | TensorFlow 2.19 / Keras 3.13 |
| Backend | Flask 2.3 |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker + Hugging Face Spaces |
| Training | Kaggle (GPU T4 x2) |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for clinical diagnosis.

---

## 👤 Author

**Ahad Ahmad**
- GitHub: [@AhadAhmad0](https://github.com/AhadAhmad0)
- Email: ahadahmad0701@gmail.com
