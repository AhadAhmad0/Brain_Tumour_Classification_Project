# Brain Tumor Classification

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)

A brain MRI classifier (4 classes) built with EfficientNetB0 transfer learning, plus a Grad-CAM investigation into why it gets gliomas wrong. It runs as a Flask app in Docker on Hugging Face Spaces.

**Live demo:** https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification

Upload a brain MRI slice and the app returns a confidence score for each of the 4 classes. Treat those scores with care. As the section on glioma errors explains, the model sometimes looks at the wrong part of the image, so a high confidence does not mean it found the tumor.

---

## Results

| Metric | Score |
|--------|-------|
| Test accuracy | 87.00% |
| Precision (weighted) | 87.30% |
| Recall (weighted) | 87.00% |
| F1 (weighted) | 86.60% |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Glioma | 93% | 72% | 81% | 400 |
| Meningioma | 82% | 78% | 80% | 400 |
| No Tumor | 86% | 99% | 92% | 400 |
| Pituitary | 87% | 99% | 93% | 400 |

The overall number hides one problem. Glioma recall is 72%, so the model misses more than a quarter of gliomas. Most of them are labeled meningioma.

<img width="1111" height="882" alt="Confusion matrix on the test set" src="https://github.com/user-attachments/assets/b2fa6a9f-9bdf-4251-9df4-3ecc80132e87" />

---

## Why the model misses gliomas

I looked for the cause before trying to fix it.

**What I ruled out.** Class imbalance and preprocessing bugs. Neither explained the glioma errors.

**What I found.** Grad-CAM heatmaps showed the model attending to the skull border and the background, not the tumor tissue. This is called shortcut learning. The model picks up patterns that happen to match the labels in this dataset, and those patterns have nothing to do with the tumor itself. It scores well on the test set because the test set has the same shortcuts.

**What I tried.** Three fixes, one at a time:

- Class-weighted loss
- Contour-based cropping to remove the background
- Cosine-decay learning rate schedule

All three scored lower than the original model, so the original is the one deployed.

**What I think would work.** Training with tumor masks, so the model is pushed to learn from the tumor region. This needs pixel-level annotations, and I have not tried it.

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar. Classes: Glioma, Meningioma, No Tumor, Pituitary.

- Training: 5,712 images
- Test: 1,600 images, 400 per class

---

## Model

```
EfficientNetB0 (ImageNet weights)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(256, relu)
    └── Dropout(0.4)
    └── Dense(128, relu)
    └── Dropout(0.3)
    └── Dense(4, softmax)
```

Training ran in two phases:

- Phase 1: base model frozen, 10 epochs, learning rate 1e-3
- Phase 2: last 30 layers of the base model unfrozen, 8 epochs, learning rate 1e-5
- EarlyStopping and ReduceLROnPlateau callbacks
- No manual rescaling, because EfficientNetB0 expects raw 0 to 255 pixel values and handles normalization inside the model

<img width="2085" height="731" alt="Training and validation curves" src="https://github.com/user-attachments/assets/f3999d00-d189-44be-ba1d-59b80b1a1c21" />

---

## Deployment

The app runs on Hugging Face Spaces in a Docker container.

```
app.py                        # Flask backend
templates/index.html          # Frontend
brain_tumor_classifier.h5     # Trained model (~32 MB)
Dockerfile
requirements.txt
```

The model file is stored on Hugging Face Spaces with Git LFS. It is not in this GitHub repository.

## Run locally

```bash
git clone https://github.com/AhadAhmad0/brain-tumor-classification
cd brain-tumor-classification

pip install -r requirements.txt

# Download brain_tumor_classifier.h5 from the Hugging Face Space
# and put it in the project root:
# https://huggingface.co/spaces/AhadAhmad0/Brain-Tumor-Classification

python app.py
```

## Tech stack

- Model: EfficientNetB0 (transfer learning)
- Framework: TensorFlow 2.19, Keras 3.13
- Backend: Flask 2.3
- Frontend: HTML, CSS, JavaScript
- Deployment: Docker, Hugging Face Spaces
- Training: Kaggle (GPU T4 x2)

## Disclaimer

This project is for learning and research. It is not a medical device and must not be used to diagnose anyone.

## Author

Ahad Ahmad
- GitHub: [@AhadAhmad0](https://github.com/AhadAhmad0)
- LinkedIn: [linkedin.com/in/ahadahmad7](https://linkedin.com/in/ahadahmad7/)
- Email: ahadahmad0701@gmail.com
