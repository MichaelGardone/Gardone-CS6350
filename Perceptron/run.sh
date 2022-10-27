#!/bin/bash

if [[ -d "Perceptron" ]]
then

    python3 Perceptron/spercep.py

    python3 Perceptron/vpercep.py > Perceptron/output/vp_performance.txt

    python3 Perceptron/apercep.py
else
    python3 Perceptron/spercep.py

    python3 Perceptron/vpercep.py > output/vp_performance.txt

    python3 Perceptron/apercep.py
fi