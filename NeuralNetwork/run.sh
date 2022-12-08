#!/bin/bash

if [[ -d "NeuralNetwork" ]]
then
    printf "*********************** Starting custom implementation! ***********************"
    python3 NeuralNetwork/nn.p
    printf "*********************** Finished custom implementation! ***********************"
    printf "*********************** Starting PyTorch implementation! ***********************"
    python3 NeuralNetwork/pytorch_nn.py
    printf "*********************** Finished PyTorch implementation! ***********************"
else
    printf "*********************** Starting custom implementation! ***********************"
    python3 nn.py
    printf "*********************** Finished custom implementation! ***********************"
    printf "*********************** Starting PyTorch implementation! ***********************"
    python3 pytorch_nn.py
    printf "*********************** Finished PyTorch implementation! ***********************"
fi
