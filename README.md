## Overview

This Project is divided into two main tasks:

### 🧠 Task 1: Speaker Verification and Identification
- Evaluated speaker verification performance using Microsoft's pre-trained **UniSpeechSatForXVector** model.
- Fine-tuned the model on VoxCeleb2 using:
  - **LoRA (Low-Rank Adaptation)** via PEFT
  - **ArcFace Loss** for better embedding separation
- Evaluated on **VoxCeleb1 (cleaned)** dataset.
- Key metrics:
  - Equal Error Rate (EER)
  - TAR@1%FAR
  - Speaker Identification Accuracy

📈 **Results after fine-tuning:**
- EER reduced from **42.30% → 9.50%**
- TAR@1%FAR improved to **~65.40%**
- Identification Accuracy improved to **~84.68%**

---

### 🗣️ Task 2: Language Identification using MFCC
- Extracted **Mel-Frequency Cepstral Coefficients (MFCC)** features from audio samples in 10 Indian languages.
- Visualized MFCC spectrograms for Gujarati, Bengali, and Tamil.
- Performed statistical analysis on mean and variance of MFCCs.
- Built a **Random Forest classifier** for language identification.

📊 **Classifier Accuracy:** ~61%  
📌 Kannada and Marathi achieved the highest performance.

---

## Files Included
- `Code.ipynb`: Jupyter Notebook with all code for both tasks.
- `Report.pdf`: Detailed report describing the methodology, results, and analysis.
- `README.md`: This file.

---

## Dependencies
- `torch`, `transformers`, `peft`, `torchaudio`
- `librosa`, `numpy`, `matplotlib`, `scikit-learn`

---

## Instructions
Open `Code.ipynb` and run the cells in order. Make sure to adjust any paths if running locally instead of Kaggle.

---
