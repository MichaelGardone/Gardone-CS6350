# python/native OS
import os

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from Bagging import Bagging

# Columns
ATTRIBUTES = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
TYPES = { "age":int, "job":str, "marital":str, "education":str, "default":str, "balance":int, "housing":str, "loan":str, "contact":str, "day":int, "month":str, "duration":int, "campaign":int, "pdays":int, "previous":int, "poutcome":str, "y":str }

FEATURES = {
    "age":          [-1, 1],
    "job":          ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    "marital":      ["married","divorced","single"],
    "education":    ["unknown","secondary","primary","tertiary"],
    "default":      ["yes","no"],
    "balance":      [-1, 1],
    "housing":      ["yes","no"],
    "loan":         ["yes","no"],
    "contact":      ["unknown","telephone","cellular"],
    "day":          [-1, 1],
    "month":        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration":     [-1, 1],
    "campaign":     [-1, 1],
    "pdays":        [-1, 1],
    "previous":     [-1, 1],
    "poutcome":     ["unknown","other","failure","success"]
}

T = 500
M = 0.5 # M is a % of all samples
DEBUG = False

save = "output/results/" if os.path.isfile("../data/bank-2/test.csv") else "EnsembleLearning/output/results/"

# making paths more resiliant so I don't have to scramble like in HW1
# bank_train = "../data/bank-2/train-subset.csv" if os.path.isfile("../data/bank-2/train-subset.csv") else "data/bank-2/train-subset.csv"
bank_train = "../data/bank-2/train.csv" if os.path.isfile("../data/bank-2/train.csv") else "data/bank-2/train.csv"
bank_test = "../data/bank-2/test.csv" if os.path.isfile("../data/bank-2/test.csv") else "data/bank-2/test.csv"

def parse_train():
    global ATTRIBUTES, TYPES
    training = pandas.read_csv(bank_train, names=ATTRIBUTES, dtype=TYPES)

    # turn numerics into a yes/no binary/get the media
    for key in TYPES:
        if TYPES[key] == int:
            median = training[key].median()
            training[key] = training[key].apply(lambda val: -1 if val < median else 1)
    
    # turn all y-labels into -1 or 1
    training["y"] = training["y"].apply(lambda val: -1 if val=="no" else 1)

    return training

def parse_test():
    global ATTRIBUTES, TYPES
    testing = pandas.read_csv(bank_test, names=ATTRIBUTES, dtype=TYPES)
    # turn numerics into a yes/no binary/get the media
    for key in TYPES:
        if TYPES[key] == int:
            median = testing[key].median()
            testing[key] = testing[key].apply(lambda val: -1 if val < median else 1)
    
    # turn all y-labels into -1 or 1
    testing["y"] = testing["y"].apply(lambda val: -1 if val=="no" else 1)

    return testing

def main():
    global FEATURES, T, DEBUG, M

    print("=== Beginning bagging tests! ===")

    tr = parse_train()
    training_length = len(tr.index)
    sample_count = int(M * training_length) # Subcount
    
    # add the weight column to the end of the training data
    tr["weight"] = 1 # in bagging, it doesn't matter the weight, but it needs to be uniform

    te = parse_test()
    testing_length = len(te.index)

    # label, sample_count, gain=gain.EntropyGain()
    bagging_model = Bagging.Bagging("y", sample_count, debug=DEBUG)

    steps = [i for i in range(T)]
    train_error_perc = []
    train_error_total = []
    test_error_perc = []
    test_error_total = []

    # columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"]
    overall = pandas.DataFrame({"T":steps})
    
    # for t = 1, 2, ..., T
    for t in range(T):
        print("\t===")
        # Train
        bagging_model.single_bag(tr, FEATURES)

        # Evaluate on train
        tr_wrong = 0
        for i in range(training_length):
            if tr["y"][i] != bagging_model.classify(tr.iloc[i]):
                tr_wrong += 1
        print("\t[", t, "] TRAINING: Total Wrong:", (tr_wrong), "/", (training_length), "(", (tr_wrong / training_length * 100), "% )")
        train_error_perc.append((tr_wrong / training_length * 100))
        train_error_total.append(tr_wrong)
        
        # Evaluate on test
        te_wrong = 0
        for i in range(testing_length):
            if te["y"][i] != bagging_model.classify(te.iloc[i]):
                te_wrong += 1
        print("\t[", t, "] TESTING: Total Wrong:", (te_wrong), "/", (testing_length), "(", (te_wrong / testing_length * 100), "% )")
        test_error_perc.append(te_wrong / testing_length * 100)
        test_error_total.append(te_wrong)
        ####
    
    overall["Training Error"] = train_error_total
    overall["Training Error %"] = train_error_perc
    overall["Testing Error"] = test_error_total
    overall["Testing Error %"] = test_error_perc
    
    ### DONE! Dump to csv
    # overall.to_csv(save + "bagging/overall_performance.csv") # uncomment to output training errors into CSV, but console should be fine for turn-in

    print("=== Finished bagging tests! ===")
    return 0

######

if __name__ == "__main__":
    main()