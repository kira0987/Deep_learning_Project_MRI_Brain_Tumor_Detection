# Brain Tumor Detection in MRI Scans

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning project developed during the **2nd Year Data Science Engineering Cycle (4DS6 - Semester 2)** at **ESPRIT**. This repository features data exploration, quality assessment, image augmentation, custom CNN modeling, and fine-tuning using VGG16 Transfer Learning for multi-class brain tumor classification from MRI scans.

---

## 📌 Project Overview

Brain tumors are critical medical conditions requiring early and accurate diagnosis. Manual MRI scan inspection is time-consuming and subject to inter-observer variability. This project implements automated deep learning pipelines to detect and classify brain MRI scans into four target categories:
* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor** (Healthy scans)

---

## 📂 Dataset Details

The dataset comprises **6,726 unique MRI images** across four distinct classes after removing duplicate files:

| Category | Image Count |
| :--- | :--- |
| **Glioma** | 1,621 |
| **Meningioma** | 1,645 |
| **No Tumor** | 2,000 |
| **Pituitary** | 1,757 |
| **Total** | **7,023 (6,726 post-deduplication)** |

---

## 🔬 Exploratory Data Analysis & Quality Checks

During data exploration, several checks were executed to evaluate image structural properties:

1. **Dimensionality Analysis:** Image widths ranged between $150$–$800$ pixels and heights between $160$–$1,080$ pixels. All images were standardized via resizing to $224 \times 224$ pixels.
2. **Brightness & Contrast:** Histogram distributions indicated healthy intensity levels peaking around 40 grayscale value. High-contrast images ($255$–$259$ range) constituted the vast majority ($5,394$ images).
3. **Data Integrity & Sharpness:**
   * **Corrupted Images:** $0$ corrupted images detected.
   * **Duplicate Images:** $297$ duplicate images identified via MD5 hash comparison and deleted to prevent overfitting.
   * **Sharpness Assessment:** Evaluated using Fast Fourier Transform (FFT) magnitude spectrum summation. All log-scaled values exceeded $7.0$ ($>10^7$), confirming high edge resolution across the dataset.

---

## 🛠️ Data Preprocessing & Augmentation

The data preparation workflow consists of:
* **Deduplication:** Removal of exact cryptographic duplicates via MD5 hashing.
* **Resizing & Normalization:** Standardized image tensor shapes to $(224, 224, 3)$ with pixel normalization to range $[0, 1]$.
* **Data Augmentation:** Brightness ($\pm 10\%$) and Contrast ($\pm 10\%$) adjustments applied dynamically via PIL `ImageEnhance`.
* **Data Splitting:**
  * **Training Set:** 80%
  * **Validation Set:** 10%
  * **Test Set:** 10% ($673$ images)

---

## 🏗️ Model Architectures

### 1. Custom CNN (Trained from Scratch)
* **Architecture:** 3 Convolutional Blocks ($\text{Conv2D} \rightarrow \text{ReLU} \rightarrow \text{MaxPooling2D}$) with filter dimensions 32, 64, and 128.
* **Classifier:** `Flatten` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(4, Softmax)`.
* **Optimization:** Adam ($\alpha = 0.0001$), Sparse Categorical Crossentropy, Batch Size 32.

### 2. VGG16 (Transfer Learning)
* **Base Model:** Pre-trained VGG16 on ImageNet with frozen lower layers.
* **Fine-Tuning:** Unfrozen top 3 convolutional layers for domain adaptation.
* **Classifier Top:** `Input(224, 224, 3)` $\rightarrow$ `VGG16 Base` $\rightarrow$ `Flatten` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(4, Softmax)`.
* **Regularization:** Early Stopping with a patience of 3 based on validation loss.

---

## 📊 Results & Comparative Analysis

Evaluation executed on the holdout test set ($673$ samples):

| Metric | Custom CNN from Scratch | VGG16 Fine-Tuned |
| :--- | :---: | :---: |
| **Overall Accuracy** | **92%** | **96%** |
| **Precision (Macro Avg)** | 92% | **96%** |
| **Recall (Macro Avg)** | 92% | **96%** |
| **F1-Score (Macro Avg)** | 92% | **96%** |
| **Test Loss** | ~0.1617 | **0.1220** |

### Class-Wise F1-Score Breakdown

| Class | Custom CNN | VGG16 Pre-trained |
| :--- | :---: | :---: |
| **Glioma** | 0.90 | **0.95** |
| **Meningioma** | 0.83 | **0.93** |
| **No Tumor** | 0.98 | **0.99** |
| **Pituitary** | 0.96 | **0.97** |

> **Conclusion:** The fine-tuned VGG16 model yielded superior generalization, resolving accuracy gaps present in the custom CNN (particularly for Meningioma detection, where recall increased from $75\%$ to $96\%$).

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Deep Learning Framework:** TensorFlow, Keras
* **Computer Vision & Processing:** OpenCV (`cv2`), PIL (Pillow)
* **Data Analysis & Visualization:** NumPy, Pandas, Matplotlib, Plotly
* **Machine Learning Tools:** Scikit-Learn

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed. Clone this repository and install necessary libraries:

```bash
git clone https://github.com/kira0987/Deep_learning_Project_MRI_Brain_Tumor_Detection.git
cd Deep_learning_Project_MRI_Brain_Tumor_Detection
pip install tensorflow opencv-python matplotlib plotly pandas numpy scikit-learn pillow
```

### Data Structure

Organize your dataset in the following hierarchy before running scripts:

```text
Data/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

### Training & Evaluation

To train and evaluate the best-performing VGG16 transfer learning model:

```python
from tensorflow.keras.models import load_model

# Load fine-tuned Keras model
model = load_model("vgg16_model.keras")

# Evaluate performance on preprocessed test images
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
