# Project_VAE
This repository contains experiments with **Variational Autoencoder (VAE)** for clustering GTZAN audio data.

## Project Overview
A **Variational Autoencoder (VAE)** is a generative model that consists of:
- An **Encoder** that maps input data to a latent distribution.
- **Reparameterization** to allow backpropagation through stochastic sampling.
- A **Decoder** that reconstructs data from the latent space.

Unlike standard autoencoders, VAEs learn a **continuous and organized latent space** by combining reconstruction loss with a regularization term (KL Divergence), enabling meaningful interpolation and sampling.

## Prerequisites
### Tensorflow
TensorFlow is required to run this project. Install TensorFlow by following the [official installation guidelines](https://www.tensorflow.org/install/pip) provided by the TensorFlow team.

The guide covers platform-specific instructions as well as CPU and GPU configurations.

### Others
Install other required dependencies for the project using the following command:
```
pip install -r requirements.txt
```

## Repository Structure
```
Project_VAE/
├── dataset/             # contains datasets
├── notebook/            # contains the main .ipynb file
├── results/             # contains results of the experiments
├── src/                 # contains utility functions
├── .gitignore
├── README.md
└── requirements.txt
```

After cloning the repository, create a dataset folder in the project directory to store your data before running the code.
