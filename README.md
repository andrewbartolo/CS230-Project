# CS230-Project
CS230 Final Project (Hybrid-ES)

1. MNIST-NN.ipynb
    - The file where most experiments/hyperparameter search took place. This file was written almost entirely by me, with some minimal starter code from a CS229 homework assignment.

2. cs230-resource-model.py
    - Defines an empirical+analytical model for Hybrid-ES's runtime, memory consumption, and network bandwidth resource usage, and generates plots of the data. This file was written entirely by me.

3. convnet-h-es.py
    - The Autograd ConvNet with Assisted H-ES. This file is a modification of https://github.com/HIPS/autograd/blob/master/examples/convnet.py to work with the Assisted H-ES algorithm. Most LOC in this file are from the autograd example.
