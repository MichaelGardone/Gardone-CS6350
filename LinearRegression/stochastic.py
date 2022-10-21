# python/native OS
import os

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# my code
from StochasticGD.StochasticGD import stch_gradient_descent, lms_cost

save = "output/results/" if os.path.isfile("../data/concrete/test.csv") else "LinearRegression/output/results/"

# making paths more resiliant so I don't have to scramble like in HW1
# bank_train = "../data/bank-2/train-subset.csv" if os.path.isfile("../data/bank-2/train-subset.csv") else "data/bank-2/train-subset.csv"
concrete_train = "../data/concrete/train.csv" if os.path.isfile("../data/concrete/train.csv") else "data/concrete/train.csv"
concrete_test = "../data/concrete/test.csv" if os.path.isfile("../data/concrete/test.csv") else "data/concrete/test.csv"

def parse_train():
    training = pandas.read_csv(concrete_train, header=None)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]
    
    tr_x = numpy.empty((row_count, col_count - 1))
    tr_y = numpy.empty((row_count, 1))

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = training.iloc[i].tolist()
        tr_x[i] = numpy.array(dp[0:col_count-1])
        tr_y[i] = dp[col_count-1]
    
    return numpy.asmatrix(tr_x), numpy.asmatrix(tr_y)

def parse_test():
    testing = pandas.read_csv(concrete_test, header=None)
    
    row_count = testing.values.shape[0]
    col_count = testing.values.shape[1]
    
    te_x = numpy.empty((row_count, col_count - 1))
    te_y = numpy.empty((row_count, 1))

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = testing.iloc[i].tolist()
        te_x[i] = numpy.array(dp[0:col_count-1])
        te_y[i] = dp[col_count-1]
    
    return numpy.asmatrix(te_x), numpy.asmatrix(te_y)

def main():
    print("=== Begin stochastic gradient descent tests! ===")

    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()
    
    # answer = batch.descent_until_converge(tr_x, tr_y) # returns (weights, iteration it finished on)
    weights, cost_history, final_costs, iters = stch_gradient_descent(tr_x, tr_y, lr=1, max_iterations=10000)
    print("w found:\n", weights)
    print("learning rate:", cost_history[-1][0])
    print("Cost of Test:", lms_cost(te_x, te_y, weights))
    
    print("Cost History:")
    print("Learn Rate | Avg. Cost (if it hit infinity)")
    lrs = []
    costs = []
    for i in range(len(cost_history)):
        output = "\t"
        output += "Learn Rate " + str(cost_history[i][0]) + ": " + str(cost_history[i][1])

        if cost_history[i][2] != -1:
            output += " (Cost became incalculable at " + str(cost_history[i][2]) + ")"

        lrs.append(cost_history[i][0])
        costs.append(cost_history[i][1])

        print(output)

    iterations = [i for i in range(iters + 1)]
    output = pandas.DataFrame({"Learning Rates":lrs, "Avg Costs":costs})

    output2 = pandas.DataFrame({"Iterations":iterations, "Cost":final_costs})

    output.to_csv(save + "stoch/avg_costs.csv")
    output2.to_csv(save + "stoch/final_iter.csv")

    print("=== Finished stochastic gradient descent tests! ===")


######

if __name__ == "__main__":
    main()