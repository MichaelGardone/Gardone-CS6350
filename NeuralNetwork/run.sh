#!/bin/bash

if [[ -d "NeuralNetwork" ]]
then
    python3 NeuralNetwork/nn.py
    python3 NeuralNetwork/pytorch_nn.py
else
    python3 nn.py
    python3 pytorch_nn.py
fi
