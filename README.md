# Gardone-CS6350
Repository for CS 6350 @ the University of Utah for Fall 2022. The code presented here was written by Michael Gardone (u1000771) in the graduate section of the Machine Learning course. All READMEs are compiled into this master one, but for more specific/less cluttered please visit each folder.

***For all projects, please be sure to at least be in the main project directory (e.g. C:/Gardone-CS6350 or /usr/Desktop/Gardone-CS6350). run.sh will not function if you aren't in there at least.***

## DecisionTrees - Homework 1

[Subfolder](https://github.com/MichaelGardone/Gardone-CS6350/tree/main/DecisionTrees)

Be sure to cd into `DecisionTrees/src` before running the following commands, either in this README or the DecisionTree README.

To run part 1 of the second part of the homework (car dataset), cd to this directory and type:

```
python3 car.py
```

To run part 2 of the second part of the homework (bank dataset), cd to this directory and type:

```
python3 bank.py
```

If you'd like to not flood your console with messages, also add the '> output.txt' piping. You can also run the test dataset (play, from the class slides) to see what is produced to verify the tree being made is correct in all three heuristics. A Python rounding error occurs in Majority Error that causes Humidity to be selected over Outlook (0.xyzw5 vs 0.xyzw4, respectively).

To run both at once, please follow:

```
cd DecisionTrees/src
./run.sh
```

## EnsembleLearning/LinearRegression - Homework 2

### [EnsembleLearning](https://github.com/MichaelGardone/Gardone-CS6350/tree/main/EnsembleLearning)

[Link Directly to Results in a Nice Google Sheet](https://docs.google.com/spreadsheets/d/1Tn2OAs4lzkiOQVqYruq6UerDEG47V1w36GB4kNc5kbc/edit?usp=sharing)

## Required Libraries

- Pandas
- Numpy

## How to Run

To run AdaBoost:

```
python3 EnsembleLearning/boost.py
```

To run Bagging:

```
python3 EnsembleLearning/bagging.py
```

To run Random Forest:

```
python3 EnsembleLearning/forest.py
```

If you'd like to not flood your console with messages, also add the '> output.txt' piping. You can also run the test dataset (play, from the class slides) to see what is produced to verify the tree being made is correct in all three heuristics.

To run all without running each command individually, enter:

```
bash ./EnsembleLearning/run.sh
```

run.sh will attempt to make a conda environment and, if unavailable, will just download Pandas and Numpy. It will clean up after itself by either deleting the conda environment or removing the libraries if they were installed.

***run.sh IN THIS FILE WILL RUN ONLY THE ENSEMBLE COMPONENTS. FOLLOW LINEAR REGRESSION'S NOTES ON HOW TO RUN IT!***

### [LinearRegression](https://github.com/MichaelGardone/Gardone-CS6350/tree/main/LinearRegression)

## How to Run

To run Batch GD:

```
python3 LinearRegression/batch.py
```

To run LinearRegression/Stochastic GD:

```
python3 LinearRegression/stochastic.py
```

To run the analytical equation:

```
python3 LinearRegression/analytical.py
```

Console flooding doesn't happen with the linear regression output, so no piping is needed.

To run all without running each command individually, enter:

```
bash ./LinearRegression/run.sh
```

***run.sh IN THIS FILE WILL RUN ONLY THE LINEAR REGRESSION COMPONENTS. FOLLOW ENSEMBLE LEARNING'S NOTES ON HOW TO RUN IT!***

## Perceptron - Homework 3

[Subfolder]()

## How to Run

To run Standard Perceptron:

```
python3 Perceptron/spercep.py
```

To run Voted Perceptron (produces a lot of output):

```
python3 Perceptron/vpercep.py > Perceptron/output/vp_performance.txt
```

To run the Averaged Perceptron:

```
python3 Perceptron/apercep.py
```

To run all without running each command individually, enter:

```
bash Perceptron/run.sh
```

The bash script does not run perceptron.py, to do that please copy/paste the following:

```
python3 Perceptron/perceptron.py
```

## ???? - Homework 4

[Subfolder]()

## ???? - Homework 5

[Subfolder]()
