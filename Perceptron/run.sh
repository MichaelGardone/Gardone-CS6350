#!/bin/bash

if [[ -d "Perceptron" ]]
then

    python3 Perceptron/spercep.py

    python3 Perceptron/vpercep_save.py

    python3 Perceptron/apercep.py
else
    python3 spercep.py

    python3 vpercep_save.py

    python3 apercep.py
fi