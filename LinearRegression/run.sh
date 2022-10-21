#!/bin/bash

if [[ -d "LinearRegression" ]]
then
    printf "Running tests for LinearRegression...\n"

    printf "Running tests for Batch GD...\n"

    python3 LinearRegression/batch.py

    printf "Finished tests for Batch GD!\n\n"

    printf "Running tests for Stochastic GD...\n"

    python3 LinearRegression/stochastic.py

    printf "Finished tests for Stochastic GD!\n\n"

    printf "Running tests for analytical equation...\n"

    python3 LinearRegression/analytical.py

    printf "Finished tests for analytical equation!\n\n"

    printf "All tests finished!\n"
else
    printf "Running tests for LinearRegression...\n"

    printf "Running tests for Batch GD...\n"

    python3 batch.py

    printf "Finished tests for Batch GD!\n\n"

    printf "Running tests for Stochastic GD...\n"

    python3 stochastic.py

    printf "Finished tests for Stochastic GD!\n\n"

    printf "Running tests for analytical equation...\n"

    python3 analytical.py

    printf "Finished tests for analytical equation!\n\n"

    printf "All tests finished!\n"
fi