🧠 Brain Tumor Classification (Deep Learning + Flask):

A web-based application that predicts whether a brain MRI scan contains a tumor using deep learning.
This project demonstrates an end-to-end machine learning pipeline, including model training, optimization, and real-world deployment handling.

🚀 Live Demo:
🔗  [Brain Tumor Classification Web App](https://brain-tumour-classification-project-24.onrender.com)

📌 Project Overview:

This application allows users to:
1.📤 Upload an MRI brain image
2.🖼️ View image preview
3.🧠 Get prediction:
  ->Tumor
  ->No Tumor
  
📊 View confidence score:

The system integrates:
1.Flask (Backend)
2.HTML/CSS/JS (Frontend)
3.TensorFlow/Keras (Model Training)

🧠 Model Development Journey:

🔹 Initial Approach: VGG19 (Transfer Learning)
The project initially used a VGG19-based model with frozen layers for feature extraction.
⚠️ Challenges with VGG Model
❌ Large model size (hundreds of MB)
❌ Not suitable for GitHub upload limits
❌ Slow inference time
❌ High memory usage during deployment
❌ TensorFlow/Keras compatibility issues on cloud platforms (Render)
❌ Serialization/loading errors (e.g., layer config mismatches)
👉 Result:
Although VGG performed reasonably well, it was not practical for deployment

🔹 Final Approach: Lightweight Model (MobileNetV2)
To overcome these issues, the model was redesigned using MobileNetV2.
✅ Advantages
✔️ Small model size (~30MB)
✔️ Faster inference
✔️ Lower memory usage
✔️ Deployment-friendly
✔️ Better compatibility with cloud environments

📊 Model Performance:
Metric Value
Training Accuracy
93.6%
Validation Accuracy
88.0%
These metrics are obtained from the final lightweight model.

⚙️ Model Inference Strategy (Hybrid System):

This project uses a hybrid inference approach to ensure reliability across environments.
🔹 Primary: Model-Based Prediction
Uses final_model.h5
Performs real deep learning inference
Returns prediction + confidence
🔹 Fallback Mechanism

If the model cannot load due to:
1.dependency conflicts
2.environment limitations
3.TensorFlow compatibility issues
Then the system:
->processes the image
->uses intensity + contrast features
->returns prediction and confidence

🎯 Why Hybrid?
1.Ensures application never breaks
2.Guarantees deployment stability
3.Demonstrates real-world engineering thinking
4.Handles practical ML deployment challenges

🧪 Training Pipeline (Notebook):

1.The project includes a Jupyter Notebook covering:
2.Data preprocessing
3.Image augmentation
4.Transfer learning (MobileNetV2)
5.Training & fine-tuning
6.Model evaluation
7.Model saving

⚠️ Deployment Note:

Deep learning models often face deployment challenges due to:
->environment inconsistencies
->dependency conflicts
->hardware limitations

This project addresses those challenges using:
-> a lightweight model
-> a hybrid fallback mechanism
-> The notebook contains the complete ML workflow, while the deployed app ensures consistent usability.

💡 Key Highlights:

1.End-to-end ML pipeline
2.Lightweight optimized model
3.Flask web application
4.Hybrid inference system
5.Real-world deployment handling
6.Clean and responsive UI

🧠 Skills Demonstrated:

1.Deep Learning (CNN, Transfer Learning)
2.TensorFlow / Keras
3.Flask Backend Development
4.Model Deployment
5.Debugging & Optimization
6.Practical Problem Solving

🔮 Future Improvements:

1.Multi-class tumor classification
2.Model quantization (faster inference)
3.Docker containerization
4.API-based deployment
5.Grad-CAM visualization
