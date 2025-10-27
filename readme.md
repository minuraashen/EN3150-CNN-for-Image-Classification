# Waste Classification using Convolutional Neural Networks

A comprehensive deep learning project for automated waste classification using custom CNN architectures and transfer learning with ResNet18 and VGG16 models.

---

## 📋 Table of Contents

- [Overview](#-overview)  
- [Dataset](#-dataset)  
- [Models](#-models)  
- [Results](#-results)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Project Structure](#-project-structure)  
- [Requirements](#-requirements)  
- [Performance Comparison](#-performance-comparison)  
- [Contributing](#-contributing)  
- [License](#-license)  

---

## 🎯 Overview

This project implements and compares multiple deep learning approaches for waste classification into 9 categories. The goal is to develop an efficient model that can accurately classify different types of waste materials to support automated recycling and waste management systems.

### Waste Categories

- Cardboard  
- Food Organics  
- Glass  
- Metal  
- Miscellaneous Trash  
- Paper  
- Plastic  
- Textile Trash  
- Vegetation  

---

## 📊 Dataset

*Source:* RealWaste Dataset  

*Distribution:*  
- Training: 70%  
- Validation: 15%  
- Testing: 15%  

*Preprocessing:*  
- Image resizing to 224×224 pixels  
- Data augmentation (horizontal flip, rotation, color jitter)  
- Normalization using ImageNet statistics  

---

## 🤖 Models

### 1. Simple CNN (Custom Architecture)

A lightweight convolutional neural network with 3 convolutional blocks:

- Conv Block 1: 32 filters (3×3)  
- Conv Block 2: 64 filters (3×3)  
- Conv Block 3: 128 filters (3×3)  
- Fully Connected: 256 neurons  
- Output: 9 classes  
- Dropout: 0.5  

*Optimizers Tested:*  
- Adam (lr=0.001)  
- SGD (lr=0.01)  
- SGD with Momentum (lr=0.01, momentum=0.9)  

### 2. ResNet18 (Transfer Learning)

- Fine-tuned ResNet18 with ImageNet pre-trained weights  
- Unfrozen layers: layer4 + fully connected  
- Discriminative learning rates  
- Weight decay: 1e-4  

### 3. VGG16 (Transfer Learning)

- Fine-tuned VGG16 with ImageNet pre-trained weights  
- Unfrozen: Last convolutional block + classifier  
- Learning rate: 1e-4  
- Weight decay: 1e-4  

---

## 📈 Results

### Simple CNN Performance

| Optimizer      | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Adam           | 72.83%   | 0.749     | 0.735  | 0.737    |
| SGD            | 63.87%   | 0.682     | 0.643  | 0.650    |
| SGD + Momentum | 65.69%   | 0.664     | 0.654  | 0.655    |

### Transfer Learning Performance

| Model    | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| ResNet18 | 89.78%   | 0.906     | 0.902  | 0.903    |
| VGG16    | 87.39%   | 0.876     | 0.875  | 0.874    |

### Best Model: ResNet18 Fine-tuned

| Class               | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| Cardboard           | 0.88      | 0.94   | 0.91     | 70      |
| Food Organics       | 0.95      | 0.95   | 0.95     | 63      |
| Glass               | 0.96      | 0.88   | 0.92     | 56      |
| Metal               | 0.83      | 0.91   | 0.87     | 126     |
| Miscellaneous Trash | 0.81      | 0.78   | 0.79     | 67      |
| Paper               | 0.99      | 0.92   | 0.95     | 74      |
| Plastic             | 0.91      | 0.86   | 0.88     | 149     |
| Textile Trash       | 0.92      | 0.90   | 0.91     | 50      |
| Vegetation          | 0.91      | 0.98   | 0.94     | 59      |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+  
- CUDA-capable GPU (recommended)  
- 8GB+ RAM  

### Setup
Clone the repository

git clone https://github.com/yourusername/waste-classification-cnn.git
cd waste-classification-cnn 


Create virtual environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt



---

## 💻 Usage

### Training Models

*Train Simple CNN with different optimizers*
python train_simple_cnn.py --optimizer adam --epochs 20

*Fine-tune ResNet18*
python train_resnet.py --epochs 15

*Fine-tune VGG16*
python train_vgg.py --epochs 15

### Evaluation

Evaluate trained model
python evaluate.py --model resnet18 --checkpoint best_model.pth

### Inference

Predict single image
python predict.py --image path/to/image.jpg --model resnet18

---

## 📁 Project Structure

waste-classification-cnn/
│
├── data/
│ └── RealWaste/
│ ├── Cardboard/
│ ├── Food Organics/
│ └── ...
│
├── models/
│ ├── simple_cnn.py
│ ├── resnet_transfer.py
│ └── vgg_transfer.py
│
├── utils/
│ ├── data_loader.py
│ ├── training.py
│ └── evaluation.py
│
├── notebooks/
│ └── EN3150_Assignment_03.ipynb
│
├── checkpoints/
│ └── best_model.pth
│
├── results/
│ ├── confusion_matrices/
│ └── loss_curves/
│
├── requirements.txt
├── train.py
├── evaluate.py
├── predict.py
└── README.md


---

## 📦 Requirements

- torch>=2.0.0  
- torchvision>=0.15.0  
- numpy>=1.24.0  
- matplotlib>=3.7.0  
- seaborn>=0.12.0  
- scikit-learn>=1.2.0  
- Pillow>=9.5.0  

---

## 🔍 Key Features

- Class Imbalance Handling: Weighted loss function based on class distribution  
- Data Augmentation: Random flips, rotations, and color jittering  
- Early Stopping: Prevents overfitting with patience-based stopping  
- Learning Rate Scheduling: ReduceLROnPlateau for adaptive learning rates  
- Comprehensive Metrics: Accuracy, precision, recall, F1-score, and confusion matrices  
- Multiple Optimizers: Comparison of Adam, SGD, and SGD with momentum  
- Transfer Learning: Leverages pre-trained ImageNet weights  

---

## 🎓 Training Details

### Hyperparameters

*Simple CNN:*  
- Batch size: 32  
- Epochs: 20 (with early stopping)  
- Learning rates: 0.001 (Adam), 0.01 (SGD)  
- Patience: 5 epochs  
- Dropout: 0.5  

*Transfer Learning:*  
- Batch size: 32  
- Epochs: 15 (with early stopping)  
- Learning rates: 1e-4 to 1e-3 (discriminative)  
- Weight decay: 1e-4  
- Patience: 5 epochs  

### Training Techniques

- Weighted Cross-Entropy Loss: Addresses class imbalance  
- Discriminative Learning Rates: Different rates for different layers  
- Gradual Unfreezing: Selective layer unfreezing for transfer learning  
- Data Normalization: ImageNet statistics for transfer learning  

---

## 📊 Performance Comparison

### Key Insights

- Transfer learning significantly outperforms custom CNN (+17% accuracy improvement)  
- Adam optimizer works best for simple CNN (72.83% vs 63-65% with SGD)  
- ResNet18 achieves highest accuracy (89.78%) with excellent generalization  
- VGG16 shows strong performance (87.39%) but slightly behind ResNet18  
- Best per-class performance: Paper (99% precision), Vegetation (98% recall)  
- Most challenging class: Miscellaneous Trash (78-81% across models)  

### Confusion Matrix Analysis

- High accuracy classes: Food Organics, Paper, Vegetation (>90%)  
- Moderate confusion: Metal and Plastic items  
- Challenging separation: Miscellaneous Trash from other categories  

---

## 🔧 Future Improvements

- [ ] Implement ensemble methods  
- [ ] Add EfficientNet architecture  
- [ ] Incorporate attention mechanisms  
- [ ] Deploy as REST API  
- [ ] Create mobile application  
- [ ] Add real-time video classification  
- [ ] Implement active learning pipeline  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository  
2. Create your feature branch (git checkout -b feature/AmazingFeature)  
3. Commit your changes (git commit -m 'Add some AmazingFeature')  
4. Push to the branch (git push origin feature/AmazingFeature)  
5. Open a Pull Request  

---


## 👥 Authors

Your Name - [GitHub Profile](https://github.com/yourusername)

---
