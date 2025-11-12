# 🎥 EDL-VAR: An Effective Deep Learning Framework for Video Anomaly Recognition

This repository contains the implementation of **EDL-VAR**, a deep learning framework developed as a **Final Year Project (FYP)** by *Muhammad Shahzad* for detecting and recognizing **anomalous activities** in video sequences.  
The model integrates **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)**, and **Self-Attention mechanisms** to efficiently capture spatial and temporal patterns from video frames.

---

##  Project Overview

The goal of this project is to design an **Effective Deep Learning Framework** for anomaly recognition in videos.  
Anomalies are activities that deviate from normal patterns, such as unusual movements or actions in surveillance footage.  
This project addresses that by combining **spatial**, **temporal**, and **contextual** understanding through a hybrid deep learning approach.

**Key Features:**
- Spatial feature extraction using CNN  
- Temporal sequence learning using LSTM  
- Enhanced focus using Self-Attention mechanism  
- Classification of video clips into normal or anomalous categories  

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Extract frames from video clips
   - Resize and normalize frames
   - Prepare sequences for temporal modeling

2. **Feature Extraction (CNN)**
   - Extract visual features from frames

3. **Sequence Learning (LSTM)**
   - Capture temporal dependencies across frames

4. **Attention Mechanism**
   - Focus on key frames contributing to anomaly detection

5. **Classification**
   - Predict whether activity is *normal* or *anomalous*

---

## 🧩 Model Architecture
Video Input → CNN → LSTM → Self-Attention → Dense Layer → Output


**Model File:** `models/CNN_LSTM_ATTENTION.h5`  
**Notebook:** `notebooks/Self_Attention_8_classes.ipynb`

---

## 📊 Dataset

You can use any of the following publicly available video datasets for training and evaluation:

- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

> *Dataset files are not uploaded due to size limitations.*

---

## 🧪 Experimental Results

| Model Variant | Architecture | Accuracy | Dataset |
|----------------|---------------|-----------|----------|
| CNN + LSTM + Attention | With self-attention enhancement | 96.8% | UCF101 |
| Transfer Learning + Attention | Using VGG16 + GRU | 97.2% | UCF101 |

*(Results may vary depending on preprocessing and tuning.)*

---
## 🖥️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/EDL-VAR-Video-Anomaly-Recognition.git
cd EDL-VAR-Video-Anomaly-Recognition

pip install -r requirements.txt

jupyter notebook notebooks/Self_Attention_8_classes.ipynb

EDL-VAR-Video-Anomaly-Recognition/
│
├── models/
│   └── CNN_LSTM_ATTENTION.h5
│
├── notebooks/
│   └── Self_Attention_8_classes.ipynb
│
├── data/                   # (optional - dataset folder)
├── results/                # accuracy plots, loss graphs, outputs
├── utils/                  # helper functions
├── requirements.txt        # dependencies
└── README.md               # project overview (this file)
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
scikit-learn

Results Visualization

Include plots and visual outputs such as:

Training vs Validation Accuracy

Loss Curves

Confusion Matrix

Example Frame Detections



