printf "Running tests for EnsembleLearning...\n"

printf "Running tests for AdaBoost...\n"

python3 boost.py > output/output_boost.txt

printf "Finished tests for AdaBosst!\n\n"

printf "Running tests for Bagging...\n"

python3 bagging.py > output/output_bagging.txt

printf "Finished tests for Bagging!\n\n"

printf "Running tests for Random Forest..."

python3 forest.py > output/output_forest.txt

printf "Finished tests for Random Forest!\n\n"

printf "All tests finished!\n"
