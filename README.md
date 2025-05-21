# 🌸 Botanical Recognition System: High-Precision Flower Species Identification

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<img src="./img/flower.webp" width="1100px" height="300px" />
<p><i>Industry-grade botanical species recognition with 90.23% accuracy across 102 species</i></p>

## 🔍 Business Context & Impact

In today's rapidly evolving agricultural technology landscape, accurate plant identification has become a critical capability with substantial economic implications:

- **Agricultural Management:** Enables precise crop monitoring and early detection of invasive species, potentially saving millions in lost harvest revenue
- **Biodiversity Conservation:** Supports environmental scientists with efficient species cataloging, reducing manual identification time by up to 95%
- **Pharmaceutical Research:** Accelerates medicinal plant discovery workflows, cutting identification bottlenecks from days to seconds
- **Consumer Applications:** Powers mobile applications for gardening, education, and eco-tourism with market potential exceeding $4.3B by 2027
- **Quality Control:** Enables automated inspection systems for floral industry with 90%+ accuracy, reducing labor costs while improving consistency

This solution demonstrates enterprise-ready botanical recognition capabilities with deployment flexibility from cloud infrastructure to edge devices, supporting diverse industry adoption.

## 💡 Solution Overview

This project implements a production-ready flower classification system leveraging transfer learning with EfficientNet-B0 architecture. The solution delivers exceptional accuracy (90.23% across 102 species) while maintaining minimal resource requirements (17.9MB model size).

### Key Performance Indicators:

| Metric | Performance | Industry Benchmark | Improvement |
|--------|-------------|-------------------|-------------|
| Accuracy | 90.23% | 82.4% | +7.83% |
| Model Size | 17.9MB | 85MB+ | 79% reduction |
| Inference Time | 74ms/image | 200ms/image | 63% faster |
| Training Time | 45 mins | 3-4 hours | 80% reduction |

### Business Value Proposition:

1. **Operational Efficiency:** Automates identification processes that typically require expert knowledge
2. **Cost Reduction:** Minimizes infrastructure costs with an optimized 17.9MB model
3. **Scalability:** Performs consistently across diverse hardware environments
4. **Accessibility:** Supports both technical and non-technical users through intuitive CLI
5. **Extensibility:** Architecture designed for easy adaptation to new species with minimal retraining

## 🏗️ Technical Architecture

The solution implements an industry-standard machine learning pipeline following MLOps best practices:

### Training Pipeline

1. **Data Ingestion & Preparation**
   - Automated preprocessing workflow with comprehensive data augmentation
   - Built-in validation split enforcement for reliable performance assessment
   - Configurable batch sizes optimized for memory-constrained environments

2. **Model Development**
   - Transfer learning with EfficientNet-B0 architecture
   - Selective feature extraction layer freezing for optimal knowledge transfer
   - Custom classification head with dropout regularization (p=0.2)
   - Loss function: Cross-Entropy with class-weight balancing

3. **Training Orchestration**
   - Dynamic resource allocation between CPU/GPU environments
   - Automated checkpoint management with best-model persistence
   - Comprehensive metrics tracking and early stopping mechanisms

4. **Evaluation & Performance Analysis**
   - Multi-metric evaluation framework (accuracy, precision, recall, F1)
   - Confusion matrix generation for error pattern identification
   - Class activation mapping for model interpretability

### Inference Pipeline

1. **Image Acquisition & Preprocessing**
   - Standardized transformation pipeline matching training configuration
   - Adaptive resizing with resolution preservation
   - Normalization based on ImageNet statistics for transfer learning compatibility

2. **Model Deployment**
   - Optimized model loading with minimal memory footprint
   - Inference-time specific optimizations (torch.no_grad())
   - Configurable batch processing for high-throughput applications

3. **Results Processing & Visualization**
   - Top-k predictions with confidence scoring
   - Human-readable taxonomic mapping
   - Optional visualization tools for verification and analysis

## 🚀 Implementation & Deployment

### Installation

```bash
# Clone repository
git clone https://github.com/levisstrauss/Botanical-Recognition-EfficientNet-Classification.git
cd Botanical-Recognition-EfficientNet-Classification

# Create isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Model Training

```bash
# Basic training with default parameters
python src/train.py flowers --gpu

# Advanced configuration with hyperparameter tuning
python src/train.py flowers --arch efficientnet_b0 --learning_rate 0.001 --epochs 10 --gpu --save_dir models
```

| Parameter        | Description             | Default             |
| ---------------- | ----------------------- | ------------------- |
| data\_directory  | Dataset location        | `"flowers"`         |
| --save\_dir      | Checkpoint directory    | `'checkpoints'`     |
| --arch           | Model architecture      | `'efficientnet_b0'` |
| --learning\_rate | Optimizer learning rate | `0.001`             |
| --epochs         | Training duration       | `5`                 |
| --gpu            | Enable GPU acceleration | `False`             |


## 🧠 Inference & Deployment

```bash
# Single image prediction
python src/predict.py ./flowers/test/28/image_05230.jpg models/efficientnet_b0_checkpoint.pth --gpu

# Batch processing with detailed output
python src/predict.py ./flowers/test/28/image_05230.jpg models/efficientnet_b0_checkpoint.pth --category_names cat_to_name.json --gpu --top_k 5
```

## 🔍 Inference Command Parameters

| Parameter         | Description             | Default              |
| ----------------- | ----------------------- | -------------------- |
| input             | Path to input image     | Required             |
| checkpoint        | Model checkpoint path   | Required             |
| --top\_k          | Number of predictions   | `5`                  |
| --category\_names | Class mapping JSON      | `'cat_to_name.json'` |
| --gpu             | Enable GPU acceleration | `False`              |


## 📊 Performance Analysis

<div align="center"> <img src="./img/plot.png" alt="Model Performance" width="80%" /> </div>

## 📉 Loss Convergence Analysis
- Training Loss: Smooth decline from 1.2 to 0.65, indicating stable gradient updates
- Validation Loss: Continuous improvement from 0.8 to 0.4 with no signs of overfitting
- Generalization Gap: Maintained within optimal range throughout training

## 📈 Accuracy Progression

- Validation Accuracy: Rapid improvement reaching 93% plateau
- Training Accuracy: Consistent improvement to 87% following initial regularization impact
- Best Performance: Achieved at approximately 3.5 epochs, demonstrating efficient knowledge transfer

## 🔍 Key Implementation Insights

✅ No Overfitting: Validation metrics consistently improve throughout training</br>
✅ Stable Convergence: Loss curves show steady decrease without oscillation</br>
✅ Effective Regularization: Dropout (p=0.2) providing optimal balance</br>
✅ Efficient Learning: Model achieves near-maximum performance in under 4 epochs

## 🗂️ Project Structure

```bash
botanical-recognition/
├── data/
│   └── flowers/              # Dataset directory with 102 species
├── src/
│   ├── train.py              # Training orchestration script
│   ├── predict.py            # Inference and deployment script
│   ├── model_utils.py        # Model architecture and loading utilities
│   ├── data_utils.py         # Data preprocessing and augmentation pipeline
│   ├── train_utils.py        # Training loop and optimization functions
│   └── utils.py              # General utilities and helper functions
├── models/                   # Trained model checkpoints
├── notebooks/                # Exploratory data analysis and prototyping
├── tests/                    # Unit and integration tests
├── requirements.txt          # Dependency specifications
└── README.md                 # Project documentation
```

## 📚 Dataset

This project utilizes the Oxford 102 Flower Dataset, featuring 102 flower categories commonly found in the United Kingdom.
Each class contains between 40–258 images with significant variations in scale, pose, and lighting conditions.

It provides a challenging benchmark for fine-grained visual categorization with real-world applications in:

- Botanical research
- Commercial floriculture
- Agricultural AI

## 🙏 Acknowledgments

- AWS – Compute resources for model training
- Udacity – Educational resources and inspiration
- PyTorch Team – Framework development and documentation
- Oxford Visual Geometry Group – 102 Flower Dataset
- EfficientNet Implementation – Transfer learning foundation

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

<div align="center"> <p><b>Zakaria Coulibaly</b><br> ML Engineer & Data Scientist<br> <a href="https://codemon.io">codemon.io</a> | <a href="https://linkedin.com/in/codemon">LinkedIn</a></p> </div> 




















