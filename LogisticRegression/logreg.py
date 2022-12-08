# python/native OS
import os, copy

# 3rd party
import pandas, numpy

# original code
import LogisticRegression

bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

VARIANCES = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
# VARIANCES = [0.01, 0.1]
T = 10

def parse_train():
    training = pandas.read_csv(bankn_train, header=None)

    training[4] = training[4].apply(lambda val: -1 if val==0 else 1)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]

    tr_x = []
    tr_y = []

    for i in range(row_count):
        dp = training.iloc[i].tolist()
        
        x = copy.deepcopy(dp[0:col_count])
        x[-1] = 1
        tr_x.append(x)

        tr_y.append(dp[col_count-1])

    return numpy.array(tr_x), numpy.array(tr_y)
##

def parse_test():
    testing = pandas.read_csv(bankn_test, header=None)
    
    testing[4] = testing[4].apply(lambda val: -1 if val==0 else 1)
    
    row_count = testing.values.shape[0]
    col_count = testing.values.shape[1]
    
    te_x = []
    te_y = []

    # magic to turn it into a nice math-ready form...
    for i in range(row_count):
        dp = testing.iloc[i].tolist()
        
        x = copy.deepcopy(dp[0:col_count])
        x[-1] = 1
        te_x.append(x)
        te_y.append(dp[col_count-1])
    
    return numpy.array(te_x), numpy.array(te_y)
##

if __name__ == "__main__":
    x = numpy.array([[0.5, -1, 0.3, 1],[-1, -2, -2, 1],[1.5, 0.2, -2.5, 1]])
    y = numpy.array([[1], [-1], [1]])

    print("===== PROBLEM 4 CHECK =====")
    prob4 = LogisticRegression.LogisticRegression(len(x[0]), "map")
    prob4.train_verbose(x, y, lr=0.01, var=1, T=3)
    print("===== PROBLEM 4 CHECK =====")

    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()

    results_mle = {}
    results_map = {}
    
    combos = len(VARIANCES) * 2

    print(f"Starting testing with {combos} variations...")

    for v in VARIANCES:
        print(f"Var={v}")

        mle_reg = LogisticRegression.LogisticRegression(len(tr_x[0]))
        mle_reg.train(tr_x, tr_y, d=0.1, lr=0.01, T=T)

        preds = mle_reg.predict_all(tr_x)
        tr_wrong = 0
        for i,p in enumerate(preds):
            if p != tr_y[i]:
                tr_wrong += 1
        

        preds = mle_reg.predict_all(te_x)
        te_wrong = 0
        for i,p in enumerate(preds):
            if p != te_y[i]:
                te_wrong += 1
        
        print(f"\tMLE Training Error: {tr_wrong / 872 * 100}\%")
        print(f"\tMLE Testing Error: {te_wrong / 500 * 100}\%")

        map_reg = LogisticRegression.LogisticRegression(len(tr_x[0]), "map")
        map_reg.train(tr_x, tr_y, d=0.1, lr=0.01, var=v, T=T)

        preds = map_reg.predict_all(tr_x)
        tr_wrong = 0
        for i,p in enumerate(preds):
            if p != tr_y[i]:
                tr_wrong += 1

        preds = map_reg.predict_all(te_x)
        te_wrong = 0
        for i,p in enumerate(preds):
            if p != te_y[i]:
                te_wrong += 1
        
        print(f"\tMAP Training Error: {tr_wrong / 872 * 100}\%")
        print(f"\tMAP Testing Error: {te_wrong / 500 * 100}\%")
    ##
    print(f"Finished testing with {combos} variations!")
    ##

##
