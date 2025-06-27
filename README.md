# IITG Internship Assignments

[![Python](https://img.shields.io/badge/Python_3.9_+-3776AB?logo=python&logoColor=FF6F00)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)

> **Brief Overview:** A comprehensive collection of deep learning projects demonstrating use cases of computer vision and natural language processing. This repository showcases various model architectures, from traditional CNNs to state-of-the-art transformer models, applied to diverse datasets and classification tasks.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Projects & Datasets](#-projects--datasets)
- [🧠 Model Architectures](#-model-architectures)
- [📈 Results Summary](#-results-summary)
- [⚙️ Quick Start](#️-quick-start)
- [🚀 Future Work](#-future-work)

## 🎯 Project Overview

This repository contains four distinct deep learning projects, each addressing different aspects of machine learning:

1. **Twitter Sentiment Classification** - NLP with BERT transformers
2. **MNIST Digit Recognition** - Comparison of MLP vs CNN architectures
3. **Dog vs Cat Classification** - Transfer learning with ResNet-101
4. **CIFAR-10 Benchmark Study** - Comprehensive model comparison

## 📊 Projects & Datasets

### 1. Twitter Sentiment Classification with BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uy_ldbjPvRAsevn8nbfSOkJ-NAIP0beR?usp=sharing)

**Objective:** Classify tweets into Positive, Negative, and Neutral sentiments using BERT.

**Dataset Information:**
- **Source:** Kagglehub (`jp797498e/twitter-entity-sentiment-analysis`)
- **Size:** ~74,000 training samples, ~1,000 validation samples
- **Format:** CSV with 'Text' and 'Sentiment' columns

**Model:** Pre-trained `bert-base-uncased` with sequence classification head

**Performance:**
- **Validation Accuracy:** 97.83%
- **Validation Loss:** 0.1190

### 2. MNIST Handwritten Digit Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oNBCiY9seWZBn2klBAfwMjHt8MIxLviD?usp=sharing)

**Objective:** Compare MLP and CNN architectures for digit recognition.

**Dataset Information:**
- **Source:** MNIST (built-in Colab dataset)
- **Size:** 60,000 training images, 10,000 test images
- **Format:** 28×28 grayscale images

**Models:** 
- Multi-Layer Perceptron (MLP) with dense layers
- Convolutional Neural Network (CNN) with Conv2D layers

### 3. Dog vs Cat Classification using ResNet-101

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DVezhbUFmZGFa9BClWhSwC4ZgILWE3js?usp=sharing)

**Objective:** Binary classification of dog and cat images using transfer learning.

**Dataset Information:**
- **Source:** Microsoft Cats and Dogs Dataset (Kaggle)
- **Size:** ~25,000 images (after removing corrupted files)
- **Format:** JPEG image files

**Model:** Pre-trained ResNet-101 (ImageNet-1K) with modified final layer

### 4. CIFAR-10 Classification Benchmark Study

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mqJNx2Hq5ZPrrwBeJxHe7Bu4otjkR4wJ?usp=sharing)

**Objective:** Comprehensive comparison of multiple architectures on CIFAR-10.

**Dataset Information:**
- **Source:** CIFAR-10 (torchvision.datasets)
- **Size:** 60,000 32×32 color images (50,000 train, 10,000 test)
- **Classes:** 10 categories with 6,000 images per class

**Models:** Custom CNN, AlexNet, ResNet18, Inception V3, Custom Architecture with residual connections

## 🧠 Model Architectures

### Natural Language Processing
- **BERT (Bidirectional Encoder Representations from Transformers)**: State-of-the-art transformer model for sentiment analysis

### Computer Vision
- **Multi-Layer Perceptron (MLP)**: Basic feedforward neural network
- **Convolutional Neural Network (CNN)**: Spatial feature extraction with convolutional layers
- **ResNet-101**: Deep residual network with skip connections
- **AlexNet**: Classic CNN architecture
- **Inception V3**: Multi-scale feature extraction
- **Custom Architectures**: Novel designs with depthwise separable convolutions

### Key Features Implemented
- Transfer learning and fine-tuning
- Data augmentation techniques
- Dropout regularization
- Residual connections
- Batch normalization
- Advanced optimizers (Adam)

## 📈 Results Summary

### Performance Comparison

| Project | Model | Dataset | Accuracy | Key Metric |
|---------|-------|---------|----------|------------|
| Twitter Sentiment | BERT | Twitter Entity Sentiment | 97.83% | Validation Accuracy |
| MNIST Digits | CNN vs MLP | MNIST | 99.30% vs 98.17% | Architecture Comparison |
| Dog vs Cat | ResNet-101 | Microsoft Cats & Dogs | 99.54% | Transfer Learning Demo |
| CIFAR-10 Benchmark | Multiple Models | CIFAR-10 | See Below | Model Comparison |

### CIFAR-10 Detailed Results

| Model | Test Accuracy |
|-------|---------------|
| CNN | 86.23% |
| AlexNet | 82.91% |
| Inception | 74.89% |
| Custom Architecture | 67.59% |
| ResNet | 61.70% |
| Mean Accuracy : | 74.86% |

<table>
<tr>
<td colspan="2" align="center">

#### SAMPLE MODEL PREDICTIONS

</td>
</tr>
<tr>
<td rowspan="2" align="center">

**Dog vs Cat Classification**
![Dog Cat Results](https://raw.githubusercontent.com/ritanjit/Dog_vs_Cat_ResNet101/main/model_predictions.png)

</td>
<td align="center">

**MNIST Classification**
![MNIST Results](https://raw.githubusercontent.com/ritanjit/MNIST_Digit_Classification_MLP_CNN/main/model_predictions_CNN.png)

</td>
</tr>
<tr>
<td align="center">

**CIFAR-10 Classification**
![Performance Comparison](https://raw.githubusercontent.com/ritanjit/CIFAR-10_Classification_Models/main/Model_Comparision.png) 

</td>
</tr>
</table>

## ⚙️ Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- TensorFlow 2.0+
- transformers library (for BERT)
- Standard ML libraries (numpy, pandas, matplotlib, scikit-learn)

### Running the Projects

1. **Open any project notebook** by clicking the respective "Open in Colab" badge
2. **Set up GPU runtime** in Colab: `Runtime` → `Change runtime type` → `GPU`
3. **Run cells sequentially** - notebooks handle dataset download and preprocessing automatically
4. **Modify configurations** in the respective config sections as needed

## 🚀 Future Work

### Probable Enhancements
- **Advanced Architectures**: Vision Transformers (ViTs), EfficientNet variants
- **Hyperparameter Optimization**: Automated tuning with Optuna/Ray Tune
- **Data Augmentation**: Advanced techniques like MixUp, CutMix
- **Model Deployment**: REST APIs and web applications
- **MLOps Integration**: Experiment tracking with MLflow/Weights & Biases

### Research Directions
- Multi-modal learning combining vision and text
- Self-supervised learning approaches
- Federated learning implementations
- Neural architecture search (NAS)
- Model compression and quantization

## 📊 Technical Specifications

### Frameworks & Libraries
- **Deep Learning**: PyTorch
- **NLP**: Hugging Face Transformers, BERT
- **Computer Vision**: torchvision
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn

---

<div align="center">


**⭐ Star this repository if you found it helpful!**

**Connect with me:** [GitHub Profile](https://github.com/ritanjit)

Made with ❤️ by **Ritanjit**

*Demonstrating the power of deep learning across diverse domains*

</div>
