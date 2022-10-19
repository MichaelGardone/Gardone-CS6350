# Homework 2 - Ensemble Learning

[Link Directly to Results in a Nice Google Sheet](https://docs.google.com/spreadsheets/d/1Tn2OAs4lzkiOQVqYruq6UerDEG47V1w36GB4kNc5kbc/edit?usp=sharing)

## Required Libraries

- Pandas
- Numpy

## How to Run

To run AdaBoost:

```
python3 boost.py
```

To run Bagging:

```
python3 bagging.py
```

To run Random Forest:

```
python3 forest.py
```

If you'd like to not flood your console with messages, also add the '> output.txt' piping. You can also run the test dataset (play, from the class slides) to see what is produced to verify the tree being made is correct in all three heuristics.

To run all without running each command individually, enter:

```
./run.sh
```

run.sh will attempt to make a conda environment and, if unavailable, will just download Pandas and Numpy. It will clean up after itself by either deleting the conda environment or removing the libraries if they were installed.
