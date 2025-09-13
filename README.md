# MNIST Digit Classification Project

This project classifies handwritten digits using a convolutional neural network.

## Dataset
The MNIST dataset contains 70,000 images of handwritten digits (0-9).

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Test the model: `python test_model.py`

## Results
- **Training Accuracy**: 98.98%
- **Validation Accuracy**: 99.13% 
- **Test Accuracy**: 99.26%

## Model Architecture
- 3 Convolutional Layers
- 2 MaxPooling Layers
- 1 Fully Connected Layer
- Output Layer with Softmax Activation
