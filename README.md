# Neural Network for Steering Detection in Two-Qubit States

## Project Overview
This project aims to develop a neural network capable of assisting and expediting steering detection for two-qubit states. Specifically, it generates a compact conclusive polytope for each state, which can be used in the optimization problem described in [this paper](https://arxiv.org/abs/1808.09349).

## Repository Structure

### Dataset
The `dataset` folder contains the Julia scripts used to generate the datasets. Two different encodings are used to create the conclusive polytopes:

- **Mother Polytope (`mp`)**: Represents the conclusive polytope as a subpolytope of a larger "mother" polytope. This encoding allows the polytope to be represented as a binary vector.
- **Free Vectors (`fv`)**: Encodes the conclusive polytope using the angles in polar representation.

You can find my datasets in this google drive [folder](https://drive.google.com/drive/folders/1-1iiPEfsrdmM2qLAyA0uYBEEnlUTaYbw?usp=sharing)

### Machine Learning Training
The `ML_training` folder contains the scripts for training the models and tuning hyperparameters. Two different encoding-specific architectures are used:

- **Mother Polytope Encoding**: This encoding uses a multi-label classification network.
- **Free Vectors Encoding**: This encoding uses a regression network.

For each encoding, we experiment with two architectures:
  1. **Single Network**: Directly predicts the conclusive polytope.
  2. **Branching Network**: First predicts if the state is steerable and then generates the conclusive polytope based on this initial prediction.

---

