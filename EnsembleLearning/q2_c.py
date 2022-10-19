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

NUM_OF_TREES = 100
T = 500
M = 0.5 # M is a % of all samples

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
    global FEATURES, T, DEBUG, M, NUM_OF_TREES

    print("=== Beginning Q2, part C tests! ===")

    tr = parse_train()
    
    # add the weight column to the end of the training data
    tr["weight"] = 1 # in bagging, it doesn't matter the weight, but it needs to be uniform

    te = parse_test()
    testing_length = len(te.index)

    bag_o_bags = []
    bag_o_single_trees = []
    
    # Train all the trees
    print("Generating trees...")
    for t in range(NUM_OF_TREES):
        # Sample 1,000 examples uniformly without replacement from the training datset
        subset_training = tr.sample(n=1000)
        sample_count = int(M * len(subset_training)) # Subcount
        # Run your bagged trees learning algorithm based on the 1,000 training examples and learn 500 trees.
        bag = Bagging.Bagging("y", sample_count)
        bag.bag(subset_training, FEATURES, T)
        bag_o_bags.append(bag)
        # For comparison, pick the first tree in each run to get 100 fully expanded trees
        bag_o_single_trees.append(bag.get_hypothesis(0))

        if t+1 == 25:
            print ("25% done!")
        if t+1 == 50:
            print ("50% done!")
        if t+1 == 75:
            print ("75% done!")
        ####
    print("Finished!")

    print("Making predictions...")
    single_predictions = [] # list of list of predictions [ [1, -1, ...], [-1, -1, ...] ]
    bagged_predictions = [] # list of list of predictions [ [1, -1, ...], [-1, -1, ...] ]
    for i in range(testing_length):
        # Single tree
        predictions = []
        for t in range(NUM_OF_TREES):
            predictions.append(bag_o_single_trees[t].predict(te.iloc[i]))
        single_predictions.append(predictions)

        predictions = []
        for t in range(NUM_OF_TREES):
            predictions.append(bag_o_bags[t].classify(te.iloc[i]))
        bagged_predictions.append(predictions)
    print("Finished!")

    # calculate biases
    st_biases = []
    bagged_biases = []

    # for variance
    st_sample_mean = []
    bagged_sample_mean = []

    print("Calculating biases...")
    for i in range(testing_length):
        gt = te["y"][i] # ground truth

        # compute bias for single trees
        avg = 0
        predictions = []
        for t in range(NUM_OF_TREES):
            pred = single_predictions[i][t]
            avg += pred
            predictions.append(pred)
        avg /= NUM_OF_TREES
        st_sample_mean.append(avg)
        internal = avg - gt
        bias = internal * internal
        st_biases.append(bias)

        # compute bias for bagged trees
        avg = 0
        predictions = []
        for t in range(NUM_OF_TREES):
            pred = bagged_predictions[i][t]
            avg += pred
            predictions.append(pred)
        avg /= NUM_OF_TREES
        bagged_sample_mean.append(avg)
        internal = avg - gt
        bias = internal * internal
        bagged_biases.append(bias)
    print("Finished!")

    # calculate variances
    st_variances = []
    bt_variances = []
    infer_pov = 1 / (testing_length - 1)
    print("Calculating variances...")
    for i in range(testing_length):
        # compute variance for single trees
        aggregator = 0
        for x in single_predictions[i]:
            aggregator += (x - st_sample_mean[i]) * (x - st_sample_mean[i])
        s_sq = infer_pov * aggregator
        st_variances.append(s_sq)

        # compute variance for single trees
        aggregator = 0
        for x in bagged_predictions[i]:
            aggregator += (x - bagged_sample_mean[i]) * (x - bagged_sample_mean[i])
        s_sq = infer_pov * aggregator
        bt_variances.append(s_sq)
    print("Finished!")

    st_avg_bias = sum(st_biases) / len(st_biases)
    st_avg_var = sum(st_variances) / len(st_variances)
    bt_avg_bias = sum(bagged_biases) / len(bagged_biases)
    bt_avg_var = sum(bt_variances) / len(bt_variances)

    print("Average Single Tree Bias:", st_avg_bias)
    print("Average Single Tree Variance:",st_avg_var)
    print("Average Bagged Bias:",bt_avg_bias)
    print("Average Bagged Variance:",bt_avg_var)

    print("=== Finished Q2, part C tests! ===")
    
    ### DONE! Dump to csv
    # overall.to_csv(save + "bagging/overall_performance.csv")

    return 0

######

if __name__ == "__main__":
    main()