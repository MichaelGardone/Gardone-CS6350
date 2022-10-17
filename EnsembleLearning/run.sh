printf "Running tests for AdaBoost..."

python3 boost.py > output/output_boost.txt

printf "Finished tests for AdaBosst!\n\n"

printf "Running tests for Bagging..."

python3 bagging.py > output/output_bagging.txt

printf "Finished tests for Bagging!\n\n"

printf ""

# python3 forest.py > output/output_forest.txt

printf ""

printf "All tests finished!\n"
