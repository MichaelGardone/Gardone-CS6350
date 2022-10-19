# python/native OS
import os

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from BatchGD import BatchGD

# Columns
ATTRIBUTES = [ "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "output" ]
TYPES = { "Cement":float, "Slag":float, "Fly ash":float, "Water":float, "SP":float, "Coarse Aggr":float, "Fine Aggr":float, "output":float }

save = "output/results/" if os.path.isfile("../data/concrete/test.csv") else "LinearRegression/output/results/"

# making paths more resiliant so I don't have to scramble like in HW1
# bank_train = "../data/bank-2/train-subset.csv" if os.path.isfile("../data/bank-2/train-subset.csv") else "data/bank-2/train-subset.csv"
concrete_train = "../data/concrete/train.csv" if os.path.isfile("../data/concrete/train.csv") else "data/concrete/train.csv"
concrete_test = "../data/concrete/test.csv" if os.path.isfile("../data/concrete/test.csv") else "data/concrete/test.csv"

def parse_train():
    training = pandas.read_csv(concrete_train, header=None)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]
    
    tr_x = []
    tr_y = []

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = training.iloc[i].tolist()
        tr_x.append(dp[0:col_count-1])
        tr_y.append(dp[col_count-1])
    
    return (tr_x, tr_y, col_count)

def parse_test():
    testing = pandas.read_csv(concrete_test, header=None)
    
    row_count = testing.values.shape[0]
    col_count = testing.values.shape[1]

    te_x = []
    te_y = []

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = testing.iloc[i].tolist()
        te_x.append(dp[0:col_count-1])
        te_y.append(dp[col_count-1])
    
    return zip(te_x, te_y)

def main():
    print("=== Begin batch gradient descent tests! ===")

    tr_x, tr_y, cols = parse_train()
    te = parse_test()
    # print(tr_x)
    # print("==")
    # print(tr_y)
    # print(cols)

    # threshold=1e-5, max_iterations=500, lr=1
    batch = BatchGD.BatchGD(1e-5, 1000, 0.01)
    answer = batch.descent(tr_x, tr_y)
    # answer = batch.descent_until_converge(tr_x, tr_y) # returns (weights, iteration it finished on)

    print("w found:", answer[0], "after t =", answer[1], "iterations")
    print("learning rate:", 0.001)
    print("Cost of Test:", batch.cost_lms(te, answer[0]))
    
    cost_hist = batch.get_cost_history()
    # print(len(cost_hist))

    # for k in cost_hist:
    #     s = ""
    #     s += "\t" + str(len(cost_hist[k]))
    #     print(s)

    # now dump to output the history in a nice spreadsheet
    # answ_cost_hist = cost_hist[answer[1]]
    # iter_count = len(answ_cost_hist)

    # iters = []

    # output = pandas.DataFrame({"Iteration":iters})

    # collated_cost_history = []
    # total = 0
    # for k in cost_hist:
    #     for i in range(len(cost_hist[k])):
    #         iters.append(total)
    #         total += 1
    #         collated_cost_history.append(cost_hist[k][i])
    #         # print(k, ":", cost_hist[k][i])

    # print(len(iters))
    # print(len(collated_cost_history))
    # output["Iteration"] = iters
    # output["Cost"] = collated_cost_history

    # output.to_csv(save + "batch/performance.csv")

    print("=== Finished batch gradient descent tests! ===")


######

if __name__ == "__main__":
    main()