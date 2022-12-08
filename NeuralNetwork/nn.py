# python/native OS
import os

# 3rd party
import pandas, numpy

# original code
import NeuralNetwork, util

bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

HIDDEN_NETWORK_SIZE = [5, 10, 25, 50, 100]
# HIDDEN_NETWORK_SIZE = [5, 10]
LAYERS = [NeuralNetwork.Sigmoid(), NeuralNetwork.Sigmoid(), NeuralNetwork.Linear()]

def parse_train():
    training = pandas.read_csv(bankn_train, header=None)

    training[4] = training[4].apply(lambda val: -1 if val==0 else 1)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]

    tr_x = []
    tr_y = []

    for i in range(row_count):
        dp = training.iloc[i].tolist()
        
        x = dp[0:col_count-1]
        x[1:] = x
        x[0] = 1
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
        
        x = dp[0:col_count-1]
        x[1:] = x
        x[0] = 1
        te_x.append(x)
        te_y.append(dp[col_count-1])
    
    return numpy.array(te_x), numpy.array(te_y)
##

def check():
    global LAYERS
    x = numpy.array([1,1,1])

    widths = numpy.array([3,3,3,1])
    lw = numpy.array([[[-1,1],[-2,2],[-3,3]], [[-1,1],[-2,2],[-3,3]], [[-1],[2],[-1.5]]],dtype=object)

    nn = NeuralNetwork.NeuralNetwork(LAYERS, lw, widths)

    print("=== Forward Propagation ===")
    res = nn._forward(x)
    layer_reses = nn._results
    for l in range(len(layer_reses)):
        print("Layer #", l, ":", layer_reses[l])
    print("Neural Network would classify as:", numpy.sign(res))
    
    print("=== Backwards Propagation ===")
    gradients = nn._backward(res - 1, True)
    print(gradients)
##

def gaussian(tr_x, tr_y, te_x, te_y):
    global HIDDEN_NETWORK_SIZE, LAYERS

    tr_error = {}
    te_error = {}

    for ns in HIDDEN_NETWORK_SIZE:
        widths = [len(tr_x[0]), ns, ns, 1]

        weights = []
        for wi in range(len(widths) - 1):
            weights.append(util.rand_init(widths[wi], widths[wi+1]))
        weights = numpy.array(weights,dtype=object)

        nn = NeuralNetwork.NeuralNetwork(LAYERS, weights, widths)

        # train(self, data, classifications, d, lr, T=100)
        nn.train(tr_x, tr_y, 0.1, 0.1)

        # predict
        predictions = nn.predict_all(tr_x)
        
        wrong = 0
        for i,p in enumerate(predictions):
            if p != tr_y[i]:
                wrong += 1
        tr_error[ns] = wrong

        predictions = nn.predict_all(te_x)
        
        wrong = 0
        for i,p in enumerate(predictions):
            if p != te_y[i]:
                wrong += 1
        te_error[ns] = wrong
        print(f"NN with width {ns} finished")

    for ns in HIDDEN_NETWORK_SIZE:
        print("==\nLayer Width:", ns)
        print(f"\tTraining Error: {tr_error[ns]} ({(tr_error[ns] / len(tr_y) * 100)}%)")
        print(f"\tTesting Error: {te_error[ns]} ({(te_error[ns] / len(te_y) * 100)}%)")
    print("==")
##

def zero(tr_x, tr_y, te_x, te_y):
    global HIDDEN_NETWORK_SIZE, LAYERS
    networkSize = [5,10]

    tr_error = {}
    te_error = {}

    for ns in HIDDEN_NETWORK_SIZE:
        widths = [len(tr_x[0]), ns, ns, 1]

        weights = []
        for wi in range(len(widths) - 1):
            weights.append(util.zero_init(widths[wi], widths[wi+1]))
        weights = numpy.array(weights,dtype=object)

        nn = NeuralNetwork.NeuralNetwork(LAYERS, weights, widths)

        # train(self, data, classifications, d, lr, T=100)
        nn.train(tr_x, tr_y, 0.1, 0.1)

        # predict
        predictions = nn.predict_all(tr_x)
        
        wrong = 0
        for i,p in enumerate(predictions):
            if p != tr_y[i]:
                wrong += 1
        tr_error[ns] = wrong

        predictions = nn.predict_all(te_x)
        
        wrong = 0
        for i,p in enumerate(predictions):
            if p != te_y[i]:
                wrong += 1
        te_error[ns] = wrong
        print(f"NN with width {ns} finished")

    for ns in HIDDEN_NETWORK_SIZE:
        print("==\nLayer Width:", ns)
        print(f"\tTraining Error: {tr_error[ns]} ({(tr_error[ns] / len(tr_y) * 100)}%)")
        print(f"\tTesting Error: {te_error[ns]} ({(te_error[ns] / len(te_y) * 100)}%)")
    print("==")
##

def main():
    print("========== BACKPROPAGATION CHECK ==========")
    check()
    print("========== BACKPROPAGATION CHECK ==========")

    # print(util.zero_init(3, 3))
    # print(util.rand_init(3, 3))
    
    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()

    print("========== RAND INIT CHECK ==========")
    gaussian(tr_x, tr_y, te_x, te_y)
    print("========== RAND INIT CHECK ==========")

    print("========== ZERO INIT CHECK ==========")
    zero(tr_x, tr_y, te_x, te_y)
    print("========== ZERO INIT CHECK ==========")

    return 0
##

########

if __name__ == "__main__":
    main()
##
