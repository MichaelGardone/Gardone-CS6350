# python/native OS
import os
from pyexpat import features

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from DecisionTree import id3, gain
from AdaBoost import AdaBoost

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
    global FEATURES, T, DEBUG, ATTRIBUTES

    tr = parse_train()
    training_length = len(tr.index)
    
    # add the weight column to the end
    tr["weight"] = 1 / training_length

    actual_labels = tr["y"].tolist()

    # te = parse_test()
    # testing_length = len(te)
    
    print("Begin entropy testing...")
    entropy = gain.EntropyGain()
    gen = id3.EID3("y", entropy)
    sroot = gen.generate_stump(tr, FEATURES)
    troot = gen.generate_tree(tr, FEATURES)
    
    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != sroot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Stump] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != troot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Tree] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")
    print("End entropy testing...")

    print("Begin ME testing...")
    me = gain.MajorityError()
    gen = id3.EID3("y", me)
    sroot = gen.generate_stump(tr, FEATURES)
    troot = gen.generate_tree(tr, FEATURES)

    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != sroot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Stump] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != troot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Tree] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")
    print("End ME testing...")

    print("Begin GI testing...")
    gi = gain.MajorityError()
    gen = id3.EID3("y", gi)
    sroot = gen.generate_stump(tr, FEATURES)
    troot = gen.generate_tree(tr, FEATURES)
    
    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != sroot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Stump] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")

    wrong = 0
    for i in range(training_length):
        if tr["y"][i] != troot.predict(tr.iloc[i]):
            wrong += 1
    print("\t[Tree] Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")
    print("End GI testing...")

######

if __name__ == "__main__":
    main()