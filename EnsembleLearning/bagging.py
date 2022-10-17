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

T = 100
M = 0.5 # M is a % of all samples
DEBUG = True

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

    tr = parse_train()
    training_length = len(tr.index)
    sample_count = int(M * training_length) # Subcount
    
    # add the weight column to the end of the training data
    tr["weight"] = 1 # in bagging, it doesn't matter the weight, but it needs to be uniform

    # te = parse_test()
    # testing_length = len(te)

    # label, sample_count, gain=gain.EntropyGain()
    bagging_model = Bagging.Bagging("y", sample_count, debug=DEBUG)

    # print("iter 1")
    # bagging_model.single_bag(tr, FEATURES)

    # wrong = 0
    # for i in range(training_length):
    #     if tr["y"][i] != bagging_model.classify(tr.iloc[i]):
    #         wrong += 1
    # print("Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

    # bagging_model.print_hypotheses()

    # print()

    # print("iter 2")
    # bagging_model.single_bag(tr, FEATURES)
    # wrong = 0
    # for i in range(training_length):
    #     if tr["y"][i] != bagging_model.classify(tr.iloc[i]):
    #         wrong += 1
    # print("Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

    # bagging_model.print_hypotheses()
    
    # for t = 1, 2, ..., T
    for t in range(T):
        # Train
        bagging_model.single_bag(tr, FEATURES)

        # Evaluate on train
        wrong = 0
        for i in range(training_length):
            if tr["y"][i] != bagging_model.classify(tr.iloc[i]):
                wrong += 1
        print("[", t, "] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

        # Evaluate on test
        ####

    return 0

######

if __name__ == "__main__":
    main()