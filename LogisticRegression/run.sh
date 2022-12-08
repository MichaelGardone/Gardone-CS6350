#!/bin/bash

if [[ -d "LogisticRegression" ]]
then
    python3 LogisticRegression/logreg.py
else
    python3 logreg.py
fi
