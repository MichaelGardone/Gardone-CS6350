# python/native OS
import os

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from DecisionTree import gain
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

T = 500
DEBUG = False

# making paths more resiliant so I don't have to scramble like in HW1
# bank_train = "../data/bank-2/train-subset.csv" if os.path.isfile("../data/bank-2/train-subset.csv") else "data/bank-2/train-subset.csv"
bank_train = "../data/bank-2/train.csv" if os.path.isfile("../data/bank-2/train.csv") else "data/bank-2/train.csv"
bank_test = "../data/bank-2/test.csv" if os.path.isfile("../data/bank-2/test.csv") else "data/bank-2/test.csv"

save = "output/results/" if os.path.isfile("../data/bank-2/test.csv") else "EnsembleLearning/output/results/"

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
    global FEATURES, T, DEBUG

    print("=== Beginning boost tests! ===")

    tr = parse_train()
    training_length = len(tr.index)
    
    # add the weight column to the end of the training data
    tr["weight"] = 1 / training_length

    te = parse_test()
    testing_length = len(te.index)

    # label, sample_count, gain=gain.EntropyGain()
    boost_model = AdaBoost.AdaBoost("y", training_length, debug=DEBUG)

    actual_labels = tr["y"].tolist()

    overall = pandas.DataFrame(columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"])
    stump_results = pandas.DataFrame(columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"])
    
    # original

    # print("iter 1")
    # boost_model.single_boost(tr, FEATURES, actual_labels)

    # wrong = 0
    # for i in range(training_length):
    #     if tr["y"][i] != boost_model.classify(tr.iloc[i]):
    #         wrong += 1
    # print("Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")
    # print("Sum Check:", sum(boost_model._prevD[0]))
    # print("Error: ", boost_model._errors[0])

    # # after iteration 1

    # boost_model.print_hypotheses()

    # print()

    # print("iter 2")
    # boost_model.single_boost(tr, FEATURES, actual_labels)
    # wrong = 0
    # for i in range(training_length):
    #     if tr["y"][i] != boost_model.classify(tr.iloc[i]):
    #         wrong += 1
    # print("Total Wrong:", (wrong), "/", (training_length), "(", (wrong / training_length * 100), "% )")
    # print("Sum Check:", sum(boost_model._prevD[1]))
    # print("Error: ", boost_model._errors[1])
    
    # # after iteration 2
    # print(tr)

    # boost_model.print_hypotheses()
    
    # for t = 1, 2, ..., T
    for t in range(T):
        print("\t===")
        # Train
        boost_model.single_boost(tr, FEATURES, actual_labels)

        # Evaluate on train
        otr_wrong = 0 # overall
        str_wrong = 0 # stump
        for i in range(training_length):
            if tr["y"][i] != boost_model.classify(tr.iloc[i]):
                otr_wrong += 1
            if tr["y"][i] != boost_model.test_hypothesis_at(tr.iloc[i]):
                str_wrong += 1

        print("\t[", t, "] TRAINING: Overall Total Wrong:", (otr_wrong), "/", (training_length), "(", ((otr_wrong / training_length) * 100), "% )")
        print("\t[", t, "] TRAINING: Stump Total Wrong:", (str_wrong), "/", (training_length), "(", (str_wrong / training_length) * 100, "% )")
        # print("\tSum Check:", sum(boost_model._prevD[t]))
        # print("\tError: ", boost_model._errors[t])

        # Evaluate on test
        ote_wrong = 0 # overall
        ste_wrong = 0 # stump
        for i in range(testing_length):
            if te["y"][i] != boost_model.classify(tr.iloc[i]):
                ote_wrong += 1
            if te["y"][i] != boost_model.test_hypothesis_at(tr.iloc[i]):
                ste_wrong += 1

        print("\t[", t, "] TESTING: Overall Total Wrong:", (ote_wrong), "/", (training_length), "(", ((ote_wrong / testing_length) * 100), "% )")
        print("\t[", t, "] TESTING: Stump Total Wrong:", (ste_wrong), "/", (training_length), "(", (ste_wrong / testing_length) * 100, "% )")

        slice = [t, otr_wrong, (otr_wrong / training_length) * 100, ote_wrong, (ote_wrong / training_length) * 100]
        res_df = pandas.DataFrame(data=[slice], columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"])
        overall = pandas.concat([overall, res_df])

        slice = [t, str_wrong, (str_wrong / testing_length) * 100, ste_wrong, (ste_wrong / testing_length) * 100]
        res_df = pandas.DataFrame(data=[slice], columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"])
        stump_results = pandas.concat([stump_results, res_df])
        ####

    print("=== Finished boost tests! ===")

    ### DONE! Dump to csv
    overall.to_csv(save + "boost/overall_performance.csv")
    stump_results.to_csv(save + "boost/stump_performance.csv")

    return 0

######

if __name__ == "__main__":
    main()