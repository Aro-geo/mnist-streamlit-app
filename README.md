# 🧠 AI Tools Assignment – Full Report & Codebase
This repository contains the deliverables for the AI Tools Assignment structured into three parts: Theoretical Understanding, Practical Implementation (Scikit-learn, TensorFlow/PyTorch, spaCy), and a brief discussion on Ethics & Optimization in AI systems.
---
# 📚 Part 1: Theoretical Understanding
Included in report.pdf: 

Overview of classical machine learning vs. deep learning

When to use Scikit-learn vs. TensorFlow/PyTorch

Benefits and trade-offs of NLP pipelines like spaCy

Ethical considerations in dataset bias, model transparency, and responsible deployment

---

# 🧪 Part 2: Practical Implementation
## ✅ Task 1: Classical ML with Scikit-learn
Dataset: Iris Dataset

Model: K-Nearest Neighbors

Deliverable: classical_ml.ipynb

Includes:

Data preprocessing

Train/test split

Model fitting

Evaluation (accuracy, confusion matrix)

## 🤖 Task 2: Deep Learning with TensorFlow
Dataset: MNIST handwritten digits

Model: CNN using tensorflow.keras

Deliverable: deep_learning_mnist.ipynb

Includes:

CNN architecture

Model training with accuracy/loss curves

Evaluation on test set

Optional: Export to .keras and deployed in Streamlit

## App Screenshot 
![image](https://github.com/user-attachments/assets/25ca96a4-c4b2-4b64-908e-3e1822b4ca80)

![image](https://github.com/user-attachments/assets/806fa018-e677-4b39-b2cb-fb51e8fc3b93)



## 🗣️ Task 3: NLP with spaCy
Dataset: Amazon Reviews (or custom text samples)

Task: Named Entity Recognition (NER) and sentiment approximation

Deliverable: nlp_spacy.py

Output:

List of recognized entities (e.g., PRODUCT, ORG)

Basic sentiment labels (positive/neutral/negative)

---

#  ⚖️ Part 3: Ethics & Optimization
Discussed in report.pdf:

⚠️ Bias & Fairness: Understanding how training data skews model behavior

🔍 Transparency: The importance of explainable AI in deployment

⚙️ Optimization: Balancing model complexity with real-world constraints (e.g., latency, size)

🧩 Tools: Model quantization, pruning, using lighter backbones

---

# 📁 Project Structure
bash
Copy
Edit
├── classical_ml.ipynb
├── deep_learning_mnist.ipynb
├── nlp_spacy.py
├── mnist_app.py                #
├── mnist_model.keras
├── requirements.txt
├── README.md
└── report.pdf
✅ Setup Instructions
bash
Copy
Edit
pip install -r requirements.txt
