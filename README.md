# Restricted Boltzmann Machine for Iris Dataset

## Overview

This project implements a Restricted Boltzmann Machine (RBM) for the Iris dataset. The Iris dataset is a classic dataset containing measurements of iris plants' leaf lengths and widths, along with their corresponding species. The goal is to classify the variety of iris plants based on their leaf measurements using RBM, a type of neural network suitable for unsupervised learning tasks.

## Implementation Details

### Dataset
- The Iris dataset consists of measurements of sepal length, sepal width, petal length, and petal width of iris plants, along with their corresponding species.
- Each instance in the dataset represents a flower sample with its feature values and species label.

### Restricted Boltzmann Machine (RBM)
- RBM is a type of energy-based model commonly used for unsupervised learning tasks such as feature learning and dimensionality reduction.
- In this project, the RBM will learn to extract relevant features from the Iris dataset's input features without supervision.
- The RBM model consists of visible units (input features) and hidden units, connected by weighted edges with a layer of biases.
- Training RBM involves minimizing the reconstruction error between input and reconstructed data using techniques such as Contrastive Divergence.

## Usage

1. **Data Preparation:** Load and preprocess the Iris dataset, ensuring appropriate feature scaling and encoding for RBM input.
2. **RBM Training:** Train the RBM model on the preprocessed dataset, adjusting model parameters such as the number of hidden units and learning rate.
3. **Feature Extraction:** Extract learned features from the trained RBM model to represent the input data in a lower-dimensional space.
4. **Classification:** Optionally, use the learned features as input to a classification algorithm to predict the species label of iris plants.

## Future Enhancements

- Explore hyperparameter tuning techniques to improve RBM performance.
- Investigate deep belief networks (DBNs) by stacking multiple RBM layers for more complex feature learning.
- Extend the RBM implementation to handle larger and more complex datasets.


