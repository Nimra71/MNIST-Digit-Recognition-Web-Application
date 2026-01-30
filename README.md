# MNIST-Digit-Recognition-Web-Application

A machine learning project that recognizes  digits using a trained neural network and provides an interactive web interface for real-time predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Dataset Information](#dataset-information)
- [Model Training Process](#model-training-process)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Running the Web Application](#running-the-web-application)
- [Improving Model Performance](#improving-model-performance)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Project Overview

This project builds a  digit recognition system using the MNIST dataset. A neural network model is trained to classify digits (0–9), evaluated using performance metrics, and deployed through a simple web interface where users can upload or draw digits for prediction.

## Key Features

- Handwritten digit classification (0–9)  
- Neural network training with performance tracking  
- Confusion matrix and classification report  
- Accuracy and loss visualization graphs  
- Interactive web app for real-time predictions  
- Model and scaler saving for reuse  

## Project Structure

MNIST_Digit_Recognition/
│
├── mnist_digit_recognition.py 

├── mnist_model.h5 / mnist_model.pkl

├── requirements.txt

└── README.md


## Installation & Requirements

Make sure Python 3.10+ is installed.

Install dependencies:

```bash```
pip install -r requirements.txt


Or manually:

pip install tensorflow scikit-learn numpy matplotlib joblib gradio

## Dataset Information

The project uses the MNIST dataset, which contains:

60,000 training images

10,000 testing images

28×28 grayscale handwritten digits (0–9)

Loaded automatically using:

from tensorflow.keras.datasets import mnist

## Model Training Process

Load MNIST dataset

Normalize pixel values (0–255 → 0–1)

Flatten images (28×28 → 784 features)

Standardize using StandardScaler

Train Neural Network

Save trained model and scaler

## Model Evaluation

The trained model is evaluated using:

Accuracy Score

Precision, Recall, F1-score

Confusion Matrix

These metrics measure how well the model performs on unseen data.

## Visualizations

The project includes visual outputs such as:

Training Loss vs Epochs

Training Accuracy vs Epochs

These help understand model learning and prediction behavior.

## Running the Web Application

After training and saving the model:

A browser link will appear where you can upload or draw a digit and instantly see the predicted result.

## Improving Model Performance

To increase accuracy:

Increase training epochs

Add more hidden layers

Use Convolutional Neural Networks (CNNs)

Apply data augmentation

Add Dropout layers to reduce overfitting

## Troubleshooting

Model file not found: Ensure the model file exists and re-run the training script

Wrong predictions: Use grayscale images, keep digit centered, use dark background with light digit

App not opening: Check terminal for correct URL, verify all libraries are installed

## Future Enhancements

Upgrade to CNN for higher accuracy

Add probability confidence scores

Deploy online (Hugging Face / Render / Heroku)

Improve UI design

## Author

Nimra Fatima

Machine Learning & AI Enthusiast
