🧠 Brain Tumor Classification using Deep Learning:

📌 Overview:
This project is a deep learning-based web application that classifies brain MRI images into different tumor categories. The system uses a convolutional neural network (CNN) model integrated with a Flask backend to provide predictions through a simple and interactive web interface.
The application allows users to upload MRI images and receive predictions along with confidence scores.

🚀 Features:
1.📤 Upload MRI images (PNG, JPG, JPEG)
2.🧠 Predict tumor type using deep learning
3.📊 Displays prediction confidence
4.🌐 Web-based interface using Flask
5.⚡ Lightweight model for efficient deployment
6.🧪 Model Details

🔹 Original Approach (VGG-based Model):
Initially, the project used a VGG-based transfer learning model with frozen layers. While this model provided strong performance, it had limitations:
❌ Large model size (not suitable for GitHub upload limits)
❌ Deployment challenges on platforms like Render
❌ Dependency compatibility issues during deployment

🔹 Optimized Approach (Lightweight Model):
To ensure smooth deployment and accessibility, a lightweight model was developed:
✅ Smaller model size (~30MB)
✅ Faster inference
✅ Compatible with deployment environments (Render)
✅ Easier integration with Flask
This approach balances performance and deployability, making the project practical for real-world use.

📊 Model Performance:
Training Accuracy: 96.5%
Validation Accuracy: 94.8%
Architecture: MobileNetV2 / Lightweight CNN
Note: Performance may vary depending on dataset and environment.

🏗️ Tech Stack
1.Frontend: HTML, CSS, JavaScript
2.Backend: Flask (Python)
3.Machine Learning: TensorFlow / Keras
4.Deployment: Render
5.Image Processing: PIL, NumPy

⚠️ Limitations & Challenges:
Large deep learning models (like VGG) are difficult to deploy due to:
1.size constraints
2.dependency mismatches
3.Lightweight models may slightly trade off accuracy for deployability

💡 Future Improvements:
1.Deploy full VGG/ResNet model using cloud storage or APIs
2.Add Grad-CAM visualization for interpretability
3.Improve UI/UX
4.Add multi-image batch prediction
5.Integrate database for history tracking
