# Martian-Core-Analysis

## Project Overview

This repository contains the final code for Modules 6 to 9, which involves mathematical modeling, data processing, and machine learning classification using RandomForestClassifier. The project includes planetary computations, particularly related to shadow detection on Mars, as well as Physics-Informed Neural Networks (PINNs) and anomaly detection.

## Features

- Implements mathematical functions for planetary calculations
- Utilizes machine learning with RandomForestClassifier
- Data handling with pandas
- Trigonometric computations for spatial analysis
- Incorporates PINNs for solving physics-based problems
- Detects anomalies in data using advanced statistical and machine learning techniques

## Installation

To run the code, ensure you have the required dependencies installed. You can install them using:

bash
pip install numpy pandas scikit-learn tensorflow torch

## Usage

1. Open the Jupyter Notebook (finalcode.ipynb).
2. Run the code cells sequentially.
3. Modify input parameters as necessary for analysis.
4. Observe the outputs, including any predictions made by the classifier, PINN solutions, and anomaly detections.

## Key Functions

### 1. if_shadow(epilat, epilon, tarlat, tarlon)

Calculates whether a target point is in shadow based on planetary positioning.

### 2. Machine Learning Model

- Uses RandomForestClassifier for classification.
- Splits dataset using train_test_split.
- Evaluates accuracy with accuracy_score.

### 3. PINN Implementation

- Utilizes Physics-Informed Neural Networks to solve partial differential equations (PDEs).
- Leverages deep learning frameworks like TensorFlow/PyTorch.
- Ensures model solutions align with known physics laws.

### 4. Anomaly Detection

- Detects unusual patterns in data using statistical and machine learning techniques.
- Implements methods such as K-Means Classifier, Autoencoders, and statistical z-scores.
- Can be applied to detect outliers in planetary measurements or sensor data.

### References

https://www.nature.com/articles/s41586-023-06601-8#:~:text=Indeed%2C%20Mars%27s%20core%20can%20be,Information%20Sections%207%20and%208
https://www.reuters.com/science/seismic-data-indicates-huge-underground-reservoir-liquid-water-mars-2024-08-12/
