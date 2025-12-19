# Diabetes Prediction Using Machine Learning & Keras

## Overview
This project applies traditional machine learning and deep learning techniques to predict diabetes outcomes using patient health data. Multiple models are trained, evaluated, and compared to highlight performance trade-offs, training cost, and overfitting behavior on structured tabular data.

The implementation follows concepts and lab-style workflows from the **IBM SkillShare – Deep Learning & Reinforcement Learning course**, with an emphasis on practical evaluation and responsible model selection.

---

## Dataset
- **Source:** UCI Pima Indians Diabetes Dataset  
- **Samples:** 768 patients  
- **Features:** 8 numerical medical attributes  
- **Target Variable:**  
  - `1` → Diabetes  
  - `0` → No Diabetes  

Due to the relatively small dataset size, model evaluation and overfitting are key considerations.

---

## Objectives
- Establish a strong baseline using traditional machine learning
- Build neural networks using Keras
- Compare model performance using accuracy and ROC-AUC
- Visualize loss, accuracy, and ROC curves
- Demonstrate when deep learning is not the optimal solution

---

## Models Implemented

### 1. Random Forest (Baseline)
- **Type:** Random Forest Classifier  
- **Number of Trees:** 200  

This model serves as a strong baseline for comparison against neural networks.

---

### 2. Neural Network — Single Hidden Layer
- **Architecture:**
  - Input Layer: 8 features
  - Hidden Layer: 12 neurons (Sigmoid)
  - Output Layer: 1 neuron (Sigmoid)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning Rate:** 0.003
- **Loss Function:** Binary Cross-Entropy
- **Epochs:** 200 → extended to 1,200
- **Preprocessing:** Feature scaling using StandardScaler  

---

### 3. Neural Network — Two Hidden Layers
- **Architecture:**
  - Hidden Layer 1: 6 neurons (ReLU)
  - Hidden Layer 2: 6 neurons (ReLU)
  - Output Layer: 1 neuron (Sigmoid)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning Rate:** 0.003
- **Epochs:** 1,500  

---

## Results Summary

| Model | Accuracy | ROC-AUC | Observations |
|------|----------|---------|--------------|
| Random Forest (200 trees) | ~77.6% | ~0.836 | Strong baseline, fast training, effective for tabular data |
| Neural Network (1 Hidden Layer) | ~72.9% | ~0.782 | Underperforms baseline, limited benefit from added complexity |
| Neural Network (2 Hidden Layers) | Slight improvement over Model 1 | Slight improvement over Model 1 | Marginal gains, long training time, overfitting observed |

---

## Model Evaluation
Performance is evaluated using:
- **Accuracy** for overall correctness
- **ROC Curve** to visualize classification trade-offs
- **ROC-AUC** as a threshold-independent performance metric  

ROC curves are generated for all models to enable direct comparison.

---

## Key Insights
- Traditional machine learning models can outperform neural networks on small, structured datasets.
- Deeper neural networks increase training time without guaranteed performance gains.
- Overfitting becomes evident after extended training, particularly beyond ~800 epochs.
- Model selection should be driven by data size, structure, and evaluation metrics rather than complexity alone.

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib  

---

## Learning Context
This project applies techniques and exercises from the **IBM SkillShare Deep Learning & Reinforcement Learning course** to a complete supervised learning workflow.

---

## Project Value
- Demonstrates practical model comparison
- Highlights evaluation beyond accuracy
- Shows awareness of overfitting and training dynamics
- Communicates results clearly for technical and non-technical audiences
