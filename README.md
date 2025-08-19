# Handwritten Digit Recognition Project

A machine learning project a neural network classifier, diffusion model, and GAN for handwritten digit recognition and generation.

## Project Overview

This project includes:

- **Convolutional Neural Network Classifier**
- **Diffusion Model**
- **GAN Model** (work in progress)

The classifier is based on the guide https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/.

## Project Structure

```
Digit_recognition/
├── dataset/
│   └── Train.csv                 # MNIST-style training data
├── weights/
│   ├── classifier_model.weights.h5    # Trained classifier weights
│   └── diffusion_model.weights.h5     # Trained diffusion model weights
├── generated_digits/             # Output directory for generated images
│   └── ddim_generated_*.png     # Generated digit samples
├── train_classifier.py           # Train the neural network classifier
├── train_diffusion.py            # Train the diffusion model
├── train_gan.py                  # Train the GAN model (WIP)
├── analyze_classifier.py         # Analyze classifier performance
├── analyze.py                    # General analysis utilities
└── requirements.txt              # Python dependencies
```

## Features

### 1. Neural Network Classifier
- **Architecture**: Feedforward neural network with dense layers
- **Performance**: High accuracy on MNIST-style digit recognition
- **Use Case**: Traditional supervised learning for digit classification

### 2. Diffusion Model
- **Type**: DDIM (Denoising Diffusion Implicit Models)
- **Capability**: Generates realistic handwritten digit images
- **Output**: High-quality 28x28 pixel digit images
- **Status**: Fully functional with pre-trained weights

### 3. GAN Model
- **Type**: Generative Adversarial Network
- **Status**: Work in progress - currently does not generate realistic digits
- **Architecture**: Generator + Discriminator with adversarial training

## Requirements

- Python 3.10+
- TensorFlow 2.x
- Required packages (see `requirements.txt`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aredheron/mnist_models.git
   cd mnist_models
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv digit
   source digit/bin/activate  # On Windows: digit\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Classifier

```bash
python train_classifier.py
```

This will:
- Load the training data from `dataset/Train.csv`
- Train a neural network classifier
- Save the trained model to `weights/classifier_model.weights.h5`

### Analyzing Results

```bash
python analyze_classifier.py
```

This will:
- Load the trained classifier
- Evaluate performance on test data
- Display accuracy metrics

### Training the Diffusion Model

```bash
python train_diffusion.py
```

Options:
- **Mode 0**: Train from scratch
- **Mode 1**: Continue training from existing weights
- **Mode 2**: Generate new digit images

### Training the GAN

```bash
python train_gan.py
```

**Note**: This is currently a work in progress and may not produce realistic results.

## Dataset

The project uses the MNIST dataset (`Train.csv`) with:
- **Format**: CSV with 785 columns
- **First column**: Digit labels (0-9)
- **Remaining columns**: 784 pixel values (28x28 flattened)
- **Size**: 73MB training data

## Model Details

### Classifier Architecture
```
Input (28, 28, 1) → Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

### Diffusion Model
- **Type**: DDIM with U-Net backbone
- **Input**: Random noise
- **Output**: Realistic digit images
- **Training**: Denoising process with timestep conditioning

### GAN Architecture
- **Generator**: Dense layers with LeakyReLU activation
- **Discriminator**: Dense layers with binary classification
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam with different learning rates

## Performance

- **Classifier**: High accuracy on validation set
- **Diffusion Model**: Generates realistic digit images
- **GAN**: Does not learn to generate realistic handwriting, currently under development

## Generated Examples

The diffusion model generates high-quality digit images saved in the `generated_digits/` directory:
- `ddim_generated_0.png` through `ddim_generated_9.png`

## Known Issues

The GAN model currently does not learn to generate realistic digits, likely due to the discriminator learning much faster than the generator.
