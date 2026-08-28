\# Deep Learning Framework for Automated Plant Disease Detection Using Leaf Images



\## 📌 Project Overview



Plant diseases can reduce crop quality and production. Identifying diseases manually can take time and requires experience.



This project develops a deep learning-based system that analyzes a plant leaf image and predicts whether the plant is healthy or affected by a particular disease.



The system includes image preprocessing, deep learning-based classification, model evaluation, and a simple Streamlit web interface.



\## 🎯 Objectives



\- Automatically detect plant diseases from leaf images.

\- Classify healthy and diseased plant leaves.

\- Apply image preprocessing and augmentation.

\- Evaluate the model using Accuracy, Precision, Recall and F1-Score.

\- Provide a confidence score for predictions.

\- Develop a simple web interface for image upload and prediction.



\## 📊 Dataset



The primary dataset used in the project is the PlantVillage dataset.



Our implementation contains:



\- 54,305 leaf images

\- 38 classes

\- Healthy and diseased plant categories



PlantDoc is considered as an additional dataset for future real-world generalization testing.



\## 🧠 Proposed Approach



The project follows this workflow:



```text

Plant Leaf Image

&#x20;      ↓

PlantVillage Dataset

&#x20;      ↓

Image Preprocessing

&#x20;      ↓

Resize / Normalize / Augmentation

&#x20;      ↓

Deep Learning Model

&#x20;      ↓

Disease Classification

&#x20;      ↓

Confidence Score

&#x20;      ↓

Streamlit Web Interface

