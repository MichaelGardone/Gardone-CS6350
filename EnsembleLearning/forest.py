# python/native OS
import os, random

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from RandomForest import RandomForest

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
    global FEATURES, T, DEBUG, M
    print("=== Begin random forest tests! ===")

    tr = parse_train()
    training_length = len(tr.index)
    sample_count = int(M * training_length) # Subcount
    
    # add the weight column to the end of the training data
    tr["weight"] = 1 # in forest, it doesn't matter the weight, but it needs to be uniform

    te = parse_test()
    testing_length = len(te.index)

    # g = {}
    # while len(g) < attrib_count:
    #         choice = random.choice(list(FEATURES))
    #         if choice not in g:
    #             g[choice] = FEATURES[choice]
    # print(g)

    # overall = pandas.DataFrame(columns=["T", "2 Training Error", "2 Training Error %", "2 Testing Error", "2 Testing Error %",
    #                                          "4 Training Error", "4 Training Error %", "4 Testing Error", "4 Testing Error %",
    #                                          "6 Training Error", "6 Training Error %", "6 Testing Error", "6 Testing Error %" ])
    steps = [i for i in range(T)]

    rf2_train_error_perc = []
    rf2_train_error_total = []
    rf2_test_error_perc = []
    rf2_test_error_total = []

    rf4_train_error_perc = []
    rf4_train_error_total = []
    rf4_test_error_perc = []
    rf4_test_error_total = []

    rf6_train_error_perc = []
    rf6_train_error_total = []
    rf6_test_error_perc = []
    rf6_test_error_total = []

    # columns=["T", "Training Error", "Training Error %", "Testing Error", "Testing Error %"]
    overall = pandas.DataFrame({"T":steps})

    rf2_model = RandomForest.RandomForest("y", sample_count, 2, debug=DEBUG)
    rf4_model = RandomForest.RandomForest("y", sample_count, 4, debug=DEBUG)
    rf6_model = RandomForest.RandomForest("y", sample_count, 6, debug=DEBUG)
    
    # for t = 1, 2, ..., T
    for t in range(T):
        print("\t===")
        # Train
        rf2_model.single_tree(tr, FEATURES)
        rf4_model.single_tree(tr, FEATURES)
        rf6_model.single_tree(tr, FEATURES)

        # Evaluate on train
        tr_rf2_wrong = 0
        tr_rf4_wrong = 0
        tr_rf6_wrong = 0
        for i in range(training_length):
            if tr["y"][i] != rf2_model.classify(tr.iloc[i]):
                tr_rf2_wrong += 1
            if tr["y"][i] != rf4_model.classify(tr.iloc[i]):
                tr_rf4_wrong += 1
            if tr["y"][i] != rf6_model.classify(tr.iloc[i]):
                tr_rf6_wrong += 1
            
        print("\t[", t, "| 2 Features ] TRAINING: Total Wrong:", (tr_rf2_wrong), "/", (training_length), "(", (tr_rf2_wrong / training_length * 100), "% )")
        rf2_train_error_perc.append(tr_rf2_wrong / training_length * 100)
        rf2_train_error_total.append(tr_rf2_wrong)
        print("\t[", t, "| 4 Features ] TRAINING: Total Wrong:", (tr_rf4_wrong), "/", (training_length), "(", (tr_rf4_wrong / training_length * 100), "% )")
        rf4_train_error_perc.append(tr_rf4_wrong / training_length * 100)
        rf4_train_error_total.append(tr_rf4_wrong)
        print("\t[", t, "| 6 Features ] TRAINING: Total Wrong:", (tr_rf6_wrong), "/", (training_length), "(", (tr_rf6_wrong / training_length * 100), "% )")
        rf6_train_error_perc.append(tr_rf6_wrong / training_length * 100)
        rf6_train_error_total.append(tr_rf6_wrong)

        # Evaluate on test
        te_rf2_wrong = 0
        te_rf4_wrong = 0
        te_rf6_wrong = 0
        for i in range(testing_length):
            if te["y"][i] != rf2_model.classify(te.iloc[i]):
                te_rf2_wrong += 1
            if te["y"][i] != rf4_model.classify(te.iloc[i]):
                te_rf4_wrong += 1
            if te["y"][i] != rf6_model.classify(te.iloc[i]):
                te_rf6_wrong += 1
        
        print("\t[", t, "| 2 Features ] TESTING: Total Wrong:", (te_rf2_wrong), "/", (testing_length), "(", (te_rf2_wrong / testing_length * 100), "% )")
        rf2_test_error_perc.append(te_rf2_wrong / testing_length * 100)
        rf2_test_error_total.append(te_rf2_wrong)
        print("\t[", t, "| 4 Features ] TESTING: Total Wrong:", (te_rf4_wrong), "/", (testing_length), "(", (te_rf4_wrong / testing_length * 100), "% )")
        rf4_test_error_perc.append(te_rf4_wrong / testing_length * 100)
        rf4_test_error_total.append(te_rf4_wrong)
        print("\t[", t, "| 6 Features ] TESTING: Total Wrong:", (te_rf6_wrong), "/", (testing_length), "(", (te_rf6_wrong / testing_length * 100), "% )")
        rf6_test_error_perc.append(te_rf6_wrong / testing_length * 100)
        rf6_test_error_total.append(te_rf6_wrong)
        ####
    
    # columns=["T", "2 Training Error", "2 Training Error %", "2 Testing Error", "2 Testing Error", "4 Training Error", "4 Training Error %", "4 Testing Error", "4 Testing Error %",
    #               "6 Training Error", "6 Training Error %", "6 Testing Error", "6 Testing Error %" ]
    overall["2 Training Error"] = rf2_train_error_total
    overall["2 Training Error %"] = rf2_train_error_perc
    overall["2 Testing Error"] = rf2_test_error_total
    overall["2 Testing Error %"] = rf2_test_error_perc

    overall["4 Training Error"] = rf4_train_error_total
    overall["4 Training Error %"] = rf4_train_error_perc
    overall["4 Testing Error"] = rf4_test_error_total
    overall["4 Testing Error %"] = rf4_test_error_perc

    overall["6 Training Error"] = rf6_train_error_total
    overall["6 Training Error %"] = rf6_train_error_perc
    overall["6 Testing Error"] = rf6_test_error_total
    overall["6 Testing Error %"] = rf6_test_error_perc

    ### DONE! Dump to csv
    # overall.to_csv(save + "forest/overall_performance.csv") # uncomment to output training errors into CSV, but console should be fine for turn-in

    print("=== Finished random forest tests! ===")
    return 0

######

if __name__ == "__main__":
    main()