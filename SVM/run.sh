#!/bin/bash

if [[ -d "SVM" ]]
then
    python3 SVM/svm.py > SVM/output/output.txt
else
    python3 svm.py > output/output.txt
fi
