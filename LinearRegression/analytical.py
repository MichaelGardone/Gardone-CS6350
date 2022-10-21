# python/native OS
import os

# 3rd party 
# we CAN use pandas!
import pandas, numpy

# making paths more resiliant so I don't have to scramble like in HW1
concrete_train = "../data/concrete/train.csv" if os.path.isfile("../data/concrete/train.csv") else "data/concrete/train.csv"
concrete_test = "../data/concrete/test.csv" if os.path.isfile("../data/concrete/test.csv") else "data/concrete/test.csv"

def parse_train():
    training = pandas.read_csv(concrete_train, header=None)

    col_count = training.values.shape[1]
    
    tr_x = numpy.copy(training.values)[:,[0,1,2,3,4,5,6]]

    tr_y = training.values[:,col_count - 1]
    
    return (tr_x, tr_y)

def parse_test():
    testing = pandas.read_csv(concrete_test, header=None)
    
    col_count = testing.values.shape[1]
    
    tr_x = numpy.copy(testing.values)[:,[0,1,2,3,4,5,6]]

    tr_y = testing.values[:,col_count - 1]
    
    return (tr_x, tr_y)

def main():
    print("=== Begin analytical equation tests! ===")

    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()

    xt = numpy.transpose(tr_x)
    xxt = numpy.linalg.inv(numpy.matmul(xt, tr_x))
    xy = numpy.matmul(xt, tr_y)
    w = numpy.matmul(xxt, xy)
    print("w found:", w)
    
    cost = 0.0
    for i in range(len(te_x)):
        cost += numpy.square(te_y[i] - te_x[i] @ numpy.transpose(w))
    cost *= 0.5
    print("Cost of Test:", cost)

    print("=== Finished analytical equation tests! ===")


######

if __name__ == "__main__":
    main()