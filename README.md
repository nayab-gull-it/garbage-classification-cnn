# ♻️ Garbage Classification using CNN

A deep learning project that classifies waste images into 6 categories
(cardboard, glass, metal, paper, plastic, trash) using a Convolutional
Neural Network with Transfer Learning (MobileNetV2), deployed as an
interactive Streamlit dashboard.

## 🔍 Overview

- **Dataset:** [Kaggle Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) (~2,467 images, 6 classes)
- **Model:** CNN with MobileNetV2 transfer learning
- **Test Accuracy:** ~80%
- **Deployment:** Streamlit Cloud

## 🚀 Live Demo

🔗 https://garbage-classification-using-cnn.streamlit.app/

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow

## 📂 Project Structure

```
garbage-app/
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies
├── model/
│   ├── garbage_classification_cnn.h5
│   └── class_indices.json
└── README.md
```

## ⚙️ Run Locally

```bash
git clone <your-repo-url>
cd garbage-app
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Details

- Base: MobileNetV2 (pretrained on ImageNet, fine-tuned)
- Input size: 150x150 RGB images
- Data augmentation: rotation, zoom, shift, flip
- Optimizer: Adam | Loss: Categorical Cross-Entropy

## 📈 Results

- Test Accuracy: 80.47%
- Evaluation metrics: Precision, Recall, F1-score, Confusion Matrix (see project notebook)

## 📄 Documentation

Full SRS document available in [`docs/SRS_Garbage_Classification_CNN.docx`](docs/SRS_Garbage_Classification_CNN.docx)

## 👤 Author

Built as part of a machine learning portfolio project.
