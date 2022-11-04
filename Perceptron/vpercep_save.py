# python/native OS
import os, copy

# 3rd party
import pandas, numpy

# my code
from Perceptrons.VPerceptron import VPerceptron

save = "output/results/" if os.path.isfile("../data/bank-note/test.csv") else "Perceptron/output/results/"

# making paths more resiliant so I don't have to scramble like in HW1
bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
# bankn_train = "../data/bank-note/train_sub.csv" if os.path.isfile("../data/bank-note/train_sub.csv") else "data/bank-note/train_sub.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

file_loc = "Perceptron/output/" if os.path.isfile("data/bank-note/test.csv") else "output/"

# CONSTANTS
T = 10
LR = 0.1

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
    
    vpercep = VPerceptron()
    vpercep.init_perceptron(tr_x, tr_y, r=LR, T=T)

    preds = vpercep.predict(te_x)

    # print weights
    unique = []
    weights = {}
    i = 0
    for wc in vpercep.get_weights():
        s = set(wc[0].flatten())
        if s not in unique:
            unique.append(s)
            weights[i] = 0
            i += 1
            
    for wc in vpercep.get_weights():
        weights[unique.index(set(wc[0].flatten()))] += wc[1]

    content = ""
    for key in weights:
        content += str(unique[key]) + "; " + str(weights[key])

    f = open(file_loc + "vp_output.txt", "w+")
    f.write(content)
    f.close()

    wrong = 0

    for i in range(te_y.shape[0]):
        if te_y[i,0] != preds[i]:
            wrong += 1
    
    print("Saved weights to", (file_loc + "vp_output.txt"))
    print("[VP] Incorrect Predictions: ", wrong, "/", te_y.shape[0], "(", (wrong / te_y.shape[0] * 100), "% )")

    return 0

####

if __name__ == "__main__":
    main()
