# Benchmarking Deep Learning Models for Glaucoma Detection: From CNN to Vision Transformers

A comprehensive benchmarking study comparing state-of-the-art deep learning architectures for automated glaucoma detection from fundus images. This repository contains the complete experimental pipeline, including hyperparameter tuning, K-fold cross-validation, ablation studies, and explainability analysis.

## 📋 Overview

Glaucoma is the second leading cause of irreversible blindness worldwide, affecting over 80 million people. Early detection through automated analysis of fundus images can significantly reduce vision loss. This project presents a systematic comparison of CNN, Vision Transformer, and hybrid architectures for glaucoma detection, achieving **99.76% accuracy** with the MaxViT-Tiny model.

### Key Highlights

- 🎯 **State-of-the-Art Performance**: 99.76% accuracy, 99.69% F1-score, and perfect 1.0000 AUC-ROC
- 🔬 **Comprehensive Benchmarking**: Systematic comparison of 4 model architectures (CNN, ViT, Hybrid, SSL)
- 🎛️ **Hyperparameter Optimization**: Grid search ensuring peak performance for each architecture
- 📊 **Robust Evaluation**: 5-fold cross-validation with overfitting analysis
- 🔍 **Explainability**: Grad-CAM visualizations for model interpretability
- 🧪 **Ablation Studies**: Systematic evaluation of preprocessing techniques

## 🏗️ Architecture Comparison

This study benchmarks four distinct deep learning architectures:

1. **EfficientNetV2-S + Disc Crop** - CNN baseline with domain-specific preprocessing
2. **DeiT-Small/16** - Vision Transformer baseline
3. **MaxViT-Tiny** - Hybrid CNN-ViT architecture (Best Performer)
4. **DINO SSL + Finetune** - Self-supervised learning approach

## 📊 Results Summary

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
|-------|--------------|--------------|------------|--------------|---------|
| **MaxViT-Tiny** | **99.76** | **99.82** | **99.55** | **99.69** | **1.0000** |
| EfficientNetV2-S + Disc Crop | 99.65 | 99.73 | 99.38 | 99.55 | 0.9999 |
| DeiT-Small/16 | 99.44 | 99.20 | 99.38 | 99.29 | 0.9998 |
| DINO SSL + Finetune | 81.80 | 85.75 | 63.93 | 73.25 | 0.8797 |

### Key Findings

- ✅ **MaxViT-Tiny achieves best performance** (99.76% accuracy) - hybrid architecture combining CNN and ViT benefits
- ✅ **Disc crop preprocessing is critical** - provides +0.11% improvement over baseline
- ✅ **Excellent generalization** - K-fold CV shows minimal overfitting (3.36% gap)
- ✅ **Hyperparameter tuning essential** - 0.38-0.45% improvement over default configurations

## 📁 Repository Structure

```
.
├── model_benchmarking_clean.ipynb    # Main Jupyter notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── models/                           # Trained model checkpoints
│   ├── glaucoma_model.pth           # Best model (MaxViT-Tiny)
│   └── best_model_info.json         # Model metadata
│
├── train/                            # Training dataset
│   ├── 0/                           # Normal images
│   └── 1/                           # Glaucoma images
│
├── val/                              # Validation dataset
│   ├── 0/
│   └── 1/
│
└── test/                             # Test dataset
    ├── 0/
    └── 1/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended) or Apple Silicon (MPS)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/glaucoma-detection-benchmarking.git
   cd glaucoma-detection-benchmarking
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
   - Navigate to `model_benchmarking_clean.ipynb`
 

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

The notebook follows academic publication standards (IMRaD format):

1. **Dataset Loading and Preprocessing**
   - Dataset statistics and visualization
   - Preprocessing pipeline (disc crop, augmentation)
   - Data loaders setup

2. **Model Architectures**
   - EfficientNetV2-S implementation
   - DeiT-Small implementation
   - MaxViT-Tiny implementation
   - DINO SSL implementation

3. **Hyperparameter Tuning**
   - Grid search for learning rate and weight decay
   - Optimal hyperparameter identification
   - Performance comparison across configurations

4. **Model Training**
   - Training with early stopping
   - Validation monitoring
   - Best model checkpointing

5. **K-Fold Cross-Validation**
   - 5-fold stratified cross-validation
   - Overfitting analysis
   - Generalization assessment

6. **Ablation Studies**
   - Preprocessing component evaluation
   - Disc crop impact analysis
   - Augmentation effect assessment

7. **Model Evaluation**
   - Test set performance metrics
   - Confusion matrices
   - ROC curves
   - Model comparison

8. **Explainability Analysis**
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

## 📊 Generated Outputs

The notebook generates several outputs:

- `model_comparison_results.csv` - Performance metrics for all models
- `kfold_cv_results.csv` - Cross-validation results
- `ablation_study_results.csv` - Ablation study findings
- `hyperparameter_tuning_summary.csv` - Optimal hyperparameters
- `confusion_matrices_all_models.png` - Confusion matrix visualizations
- `roc_curves_comparison.png` - ROC curve comparisons
- `gradcam_*.png` - Grad-CAM visualizations
- Model checkpoints in `models/` directory

## 🤝 Contributing

This is an academic research project. If you find this work useful, please cite:

```bibtex
@article{glaucoma2024,
  title={Benchmarking Deep Learning Models for Glaucoma Detection: From CNN to Vision Transformers},
  author={biswas.shaon@gmail.com},
}
```

## 🙏 Acknowledgments

- Datasets were collected from Kaggle
- PyTorch and timm communities
- Vision Transformer research community
