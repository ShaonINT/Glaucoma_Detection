# Benchmarking Deep Learning Models for Glaucoma Detection: From CNN to Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive data science portfolio project comparing state-of-the-art deep learning architectures for automated glaucoma detection from fundus images. This repository showcases a complete experimental pipeline including hyperparameter tuning, K-fold cross-validation, ablation studies, and explainability analysis.

## 📋 Project Overview

Glaucoma is the second leading cause of irreversible blindness worldwide, affecting over 80 million people. Early detection through automated analysis of fundus images can significantly reduce vision loss. This project presents a systematic comparison of CNN, Vision Transformer, and hybrid architectures for glaucoma detection, achieving **99.76% accuracy** with the MaxViT-Tiny model.

### 🎯 Key Highlights

- 🏆 **State-of-the-Art Performance**: 99.76% accuracy, 99.69% F1-score, and perfect 1.0000 AUC-ROC
- 🔬 **Comprehensive Benchmarking**: Systematic comparison of 4 model architectures (CNN, ViT, Hybrid, SSL)
- 🎛️ **Hyperparameter Optimization**: Grid search ensuring peak performance for each architecture
- 📊 **Robust Evaluation**: 5-fold cross-validation with overfitting analysis
- 🔍 **Explainability**: Grad-CAM visualizations for model interpretability
- 🧪 **Ablation Studies**: Systematic evaluation of preprocessing techniques
- 📝 **Academic Publication-Ready**: Structured notebook following IMRaD format

## 🏗️ Architecture Comparison

This study benchmarks four distinct deep learning architectures:

1. **EfficientNetV2-S + Disc Crop** - CNN baseline with domain-specific preprocessing
2. **DeiT-Small/16** - Vision Transformer baseline
3. **MaxViT-Tiny** - Hybrid CNN-ViT architecture (Best Performer) ⭐
4. **DINO SSL + Finetune** - Self-supervised learning approach

## 📊 Results Summary

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
|-------|--------------|--------------|------------|--------------|---------|
| **MaxViT-Tiny** | **99.76** | **99.82** | **99.55** | **99.69** | **1.0000** |
| EfficientNetV2-S + Disc Crop | 99.65 | 99.73 | 99.38 | 99.55 | 0.9999 |
| DeiT-Small/16 | 99.44 | 99.20 | 99.38 | 99.29 | 0.9998 |
| DINO SSL + Finetune | 81.80 | 85.75 | 63.93 | 73.25 | 0.8797 |

### 🔑 Key Findings

- ✅ **MaxViT-Tiny achieves best performance** (99.76% accuracy) - hybrid architecture combining CNN and ViT benefits
- ✅ **Disc crop preprocessing is critical** - provides +0.11% improvement over baseline
- ✅ **Excellent generalization** - K-fold CV shows minimal overfitting (3.36% gap)
- ✅ **Hyperparameter tuning essential** - 0.38-0.45% improvement over default configurations

## 📁 Repository Structure

```
.
├── Glaucoma_Detection_model_benchmarking.ipynb    # Main Jupyter notebook (portfolio showcase)
├── requirements.txt                               # Python dependencies
├── README.md                                      # This file
└── .gitignore                                     # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended) or Apple Silicon (MPS)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShaonINT/Glaucoma_Detection.git
   cd Glaucoma_Detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Navigate to `Glaucoma_Detection_model_benchmarking.ipynb`

3. **Run all cells:**
   - The notebook will automatically:
     - Load and preprocess the dataset
     - Perform hyperparameter tuning for each model
     - Train all models with optimal hyperparameters
     - Conduct 5-fold cross-validation
     - Perform ablation studies
     - Generate Grad-CAM visualizations
     - Create comparison visualizations and reports

## 📓 Notebook Structure

The notebook follows academic publication standards (IMRaD format) and includes:

### 1. Dataset Loading and Preprocessing
- Dataset statistics and visualization
- Preprocessing pipeline (disc crop, augmentation)
- Data loaders setup

### 2. Model Architectures
- EfficientNetV2-S implementation
- DeiT-Small implementation
- MaxViT-Tiny implementation
- DINO SSL implementation

### 3. Hyperparameter Tuning
- Grid search for learning rate and weight decay
- Optimal hyperparameter identification
- Performance comparison across configurations

### 4. Model Training
- Training with early stopping
- Validation monitoring
- Best model checkpointing

### 5. K-Fold Cross-Validation ⭐
- 5-fold stratified cross-validation
- Overfitting analysis
- Generalization assessment

### 6. Ablation Studies ⭐
- Preprocessing component evaluation
- Disc crop impact analysis
- Augmentation effect assessment

### 7. Model Evaluation
- Test set performance metrics
- Confusion matrices
- ROC curves
- Model comparison

### 8. Explainability Analysis
- Grad-CAM visualization
- Attention map generation
- Clinical relevance validation

## 🔬 Methodology Highlights

### Hyperparameter Tuning
- **Grid Search**: Learning rate [1×10⁻⁴, 5×10⁻⁵, 2×10⁻⁵] × Weight decay [1×10⁻⁴, 1×10⁻⁵]
- **Result**: All top models converged to LR=1×10⁻⁴
- **Impact**: 0.38-0.45% accuracy improvement over default configurations

### K-Fold Cross-Validation
- **Method**: 5-fold stratified cross-validation
- **Purpose**: Robust generalization assessment and overfitting detection
- **Result**: Minimal overfitting gaps (3.36-3.52%) across all models

### Ablation Study
- **Configurations**: 7 different preprocessing combinations
- **Finding**: Disc crop preprocessing provides most significant improvement (+0.11%)
- **Insight**: Aligns with clinical practice (optic disc is primary diagnostic region)

### Explainability
- **Method**: Grad-CAM visualization
- **Finding**: All models focus on optic disc region (clinically relevant)
- **Impact**: Validates model learning and enables clinical trust

## 📈 Dataset

**Data Source**: [Kaggle](https://www.kaggle.com) - Fundus Image Dataset for Glaucoma Detection

- **Total Images**: 17,242 fundus images
- **Training Set**: 8,621 images (5,293 normal, 3,328 glaucoma)
- **Validation Set**: 5,747 images (3,539 normal, 2,208 glaucoma)
- **Test Set**: 2,874 images (1,754 normal, 1,120 glaucoma)
- **Class Balance**: ~1.59:1 (Normal:Glaucoma)


## 🎯 Key Contributions

1. **First systematic comparison** of CNN, ViT, and hybrid architectures for glaucoma detection
2. **State-of-the-art performance** (99.76% accuracy) exceeding previous studies
3. **Comprehensive evaluation** including hyperparameter tuning, K-fold CV, and ablation studies
4. **Explainability integration** through Grad-CAM visualizations
5. **Reproducible benchmark** with complete code and methodology

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **timm**: Pre-trained model library
- **scikit-learn**: Evaluation metrics
- **matplotlib/seaborn**: Visualization
- **pandas**: Data analysis
- **Jupyter**: Interactive development

## 📊 Portfolio Highlights

This project demonstrates:

- ✅ **Advanced Model Architecture Knowledge**: CNN, Vision Transformers, Hybrid models
- ✅ **Rigorous Evaluation Methodology**: K-fold CV, hyperparameter tuning, ablation studies
- ✅ **Medical AI Expertise**: Domain-specific preprocessing, clinical validation
- ✅ **Explainability**: Grad-CAM visualizations for model interpretability
- ✅ **Statistical Rigor**: Overfitting analysis, generalization assessment
- ✅ **Research Communication**: Publication-ready notebook and paper


## 📧 Contact

- **GitHub**: [@ShaonINT](https://github.com/ShaonINT)
- **Repository**: [Glaucoma_Detection](https://github.com/ShaonINT/Glaucoma_Detection)

## 🙏 Acknowledgments

- **Dataset**: Fundus image dataset from [Kaggle](https://www.kaggle.com)
- **PyTorch** and **timm** communities for excellent deep learning frameworks
- **Vision Transformer** research community for architectural innovations

---

**Note**: This repository focuses on the research methodology and benchmarking notebook. The dataset and trained models are excluded due to size constraints.

**Disclaimer**: This tool is designed for research and portfolio purposes. Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.
