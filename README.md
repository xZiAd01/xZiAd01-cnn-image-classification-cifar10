# CIFAR-10 Image Classification using CNN (PyTorch)

## Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10 dataset** using **PyTorch**.  
It was developed as part of an academic Artificial Intelligence assignment and emphasizes **reproducibility**, **clear experimentation**, and **proper evaluation**.

The model is trained and evaluated inside a Jupyter Notebook with support for CPU and GPU execution.

---

## Key Features
- CNN model implemented from scratch using PyTorch
- Image preprocessing and normalization
- Automatic device selection (CPU / GPU)
- Fixed random seeds for reproducible results
- K-Fold Cross-Validation
- Training and evaluation visualization

---

## Technologies Used
- Python 3.9+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- Jupyter Notebook

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
```

---

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Running the Project
Start Jupyter Notebook:

```bash
jupyter notebook
```
Open the notebook:
```
cifar10_cnn_pytorch.ipynb
```
Run all cells in order.
The CIFAR-10 dataset will be downloaded automatically on first execution.

## Evaluation
The model is evaluated using K-Fold Cross-Validation

Metrics include:

- Classification accuracy

- Training and validation loss

- Results are displayed directly within the notebook

## Limitations
- CIFAR-10 is a small-scale dataset and may not generalize to real-world data

- CNN architecture is relatively simple

- No automated hyperparameter tuning

## Future Improvements
- Apply data augmentation techniques

- Experiment with deeper architectures (e.g., ResNet, VGG)

- Learning rate scheduling

- Hyperparameter optimization

- Add experiment tracking (TensorBoard / Weights & Biases)

## Author
Zeyad Alghamdi
Computer Science – Artificial Intelligence
King Abdulaziz University

## License
This project is intended for educational purposes only.
