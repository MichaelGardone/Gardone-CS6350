# python/native OS
import os

# 3rd party
import pandas, numpy

# my code
from Perceptrons.SPerceptron import SPerceptron
from Perceptrons.VPerceptron import VPerceptron
from Perceptrons.APerceptron import APerceptron

save = "output/results/" if os.path.isfile("../data/bank-note/test.csv") else "Perceptron/output/results/"

# making paths more resiliant so I don't have to scramble like in HW1
bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
# bankn_train = "../data/bank-note/train_sub.csv" if os.path.isfile("../data/bank-note/train_sub.csv") else "data/bank-note/train_sub.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

# CONSTANTS
T = 10
LR = 0.1
LOOPS = 100

def parse_train():
    training = pandas.read_csv(bankn_train, header=None)

    training[4] = training[4].apply(lambda val: -1 if val==0 else 1)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]
    
    tr_x = numpy.empty((row_count, col_count))
    tr_y = numpy.empty((row_count, 1))

    for i in range(row_count):
        dp = training.iloc[i].tolist()
        tr_x[i] = numpy.array(dp[0:col_count])
        tr_x[i,col_count-1] = 1
        tr_y[i] = dp[col_count-1]

    return tr_x, tr_y

def parse_test():
    testing = pandas.read_csv(bankn_test, header=None)
    
    testing[4] = testing[4].apply(lambda val: -1 if val==0 else 1)
    
    row_count = testing.values.shape[0]
    col_count = testing.values.shape[1]
    
    te_x = numpy.empty((row_count, col_count))
    te_y = numpy.empty((row_count, 1))

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = testing.iloc[i].tolist()
        te_x[i] = numpy.array(dp[0:col_count])
        te_x[i,col_count-1] = 1
        te_y[i] = dp[col_count-1]
    
    return te_x, te_y

def shuffle(size, shuffle_count):
    # Generate a random array of indices
    shuffled_indices = []

    for _ in range(shuffle_count):
        shuffled = numpy.arange(size)
        numpy.random.default_rng().shuffle(shuffled)
        shuffled_indices.append(shuffled)

    return shuffled_indices

def main():
    global T, LR, LOOPS
    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()
    
    sp_wrong = []
    vp_wrong = []
    ap_wrong = []

    for _ in range(LOOPS):
        # uniform
        shuffled_indices = shuffle(te_x.shape[0], T)

        spercep = SPerceptron(shuffled_indices)
        spercep.init_perceptron(tr_x, tr_y, r=LR, T=T)

        preds = spercep.predict(te_x)

        wrong = 0
        for i in range(te_y.shape[0]):
            if te_y[i,0] != preds[i]:
                wrong += 1
        sp_wrong.append(wrong)

        vpercep = VPerceptron(shuffled_indices)
        vpercep.init_perceptron(tr_x, tr_y, r=LR, T=T)

        # print("[VP] Weights:", vpercep.get_weights())

        preds = vpercep.predict(te_x)

        wrong = 0
        for i in range(te_y.shape[0]):
            if te_y[i,0] != preds[i]:
                wrong += 1
        vp_wrong.append(wrong)

        apercep = APerceptron(shuffled_indices)
        apercep.init_perceptron(tr_x, tr_y, r=LR, T=T)

        preds = apercep.predict(te_x)

        wrong = 0
        for i in range(te_y.shape[0]):
            if te_y[i,0] != preds[i]:
                wrong += 1
        ap_wrong.append(wrong)

    print("=== Running Information ===")
    print("# of loops:", LOOPS)
    print("T:", T)
    print("R:", LR)
    print("=== Running Information ===\n")

    print("=== Results ===")
    print("Standard Perceptron")

    worst = max(sp_wrong)
    best = min(sp_wrong)
    avg = sum(sp_wrong) / len(sp_wrong)

    print("\tWorst Performance:", worst)
    print("\tBest Performance:", best)
    print("\tAverage Performance:", avg)

    worst = max(vp_wrong)
    best = min(vp_wrong)
    avg = sum(vp_wrong) / len(vp_wrong)

    print("Voted Perceptron")
    print("\tWorst Performance:", worst)
    print("\tBest Performance:", best)
    print("\tAverage Performance:", avg)

    worst = max(ap_wrong)
    best = min(ap_wrong)
    avg = sum(ap_wrong) / len(ap_wrong)
    
    print("Average Perceptron")
    print("\tWorst Performance:", worst)
    print("\tBest Performance:", best)
    print("\tAverage Performance:", avg)
    print("=== Results ===")

    return 0

####

if __name__ == "__main__":
    main()
