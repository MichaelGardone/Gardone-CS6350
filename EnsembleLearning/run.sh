#!/bin/bash
printf "Running tests for EnsembleLearning...\n"

if [[ -d "EnsembleLearning" ]]
then
    printf "Running tests for AdaBoost... (in output/output_boost.txt)\n"

    python3 EnsembleLearning/boost.py > EnsembleLearning/output/output_boost.txt

    printf "Finished tests for AdaBosst!\n\n"

    printf "Running tests for Bagging... (in output/output_bagging.txt)\n"

    python3 EnsembleLearning/bagging.py > EnsembleLearning/output/output_bagging.txt

    printf "Finished tests for Bagging!\n\n"

    printf "Running tests for Random Forest... (in output/output_forest.txt)\n"

    python3 EnsembleLearning/forest.py > EnsembleLearning/output/output_forest.txt

    printf "Finished tests for Random Forest!\n\n"

    printf "==== These next tests are Q2 C and D, you can jump out now! ===\n"
    printf "These tests produce output to inform you of what step you're on, so these aren't piped.\n"

    printf "Running tests for Q2, part C...\n"

    python3 EnsembleLearning/q2_c.py

    printf "Finished tests for Q2, part C!\n\n"

    printf "Running tests for Q2, part E...\n"

    python3 EnsembleLearning/q2_c.py

    printf "Finished tests for Q2, part E!\n\n"
else
    printf "Running tests for AdaBoost... (in output/output_boost.txt)\n"

    python3 boost.py > output/output_boost.txt

    printf "Finished tests for AdaBosst!\n\n"

    printf "Running tests for Bagging... (in output/output_bagging.txt)\n"

    python3 bagging.py > output/output_bagging.txt

    printf "Finished tests for Bagging!\n\n"

    printf "Running tests for Random Forest... (in output/output_forest.txt)\n"

    python3 forest.py > output/output_forest.txt

    printf "Finished tests for Random Forest!\n\n"

    printf "==== These next tests are Q2 C and D, you can jump out now! ===\n"
    printf "These tests produce output to inform you of what step you're on, so these aren't piped.\n"

    printf "Running tests for Q2, part C...\n"

    python3 q2_c.py

    printf "Finished tests for Q2, part C!\n\n"

    printf "Running tests for Q2, part E...\n"

    python3 q2_c.py

    printf "Finished tests for Q2, part E!\n\n"
fi

printf "All tests finished!\n"
