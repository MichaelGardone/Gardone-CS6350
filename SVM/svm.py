# python/native OS
import os

# 3rd party
import pandas, numpy

import PrimalSVM, DualSVM, GDualSVM, util

bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

C = [100/873, 500/873, 700/873]
# C = [500/873]
T = 100
GSTART = [0.1, 0.5, 1, 5, 100]
# GSTART = [0.01]

def parse_train():
    training = pandas.read_csv(bankn_train, header=None)

    training[4] = training[4].apply(lambda val: -1 if val==0 else 1)

    row_count = training.values.shape[0]
    col_count = training.values.shape[1]

    tr_x = []
    tr_y = []

    for i in range(row_count):
        dp = training.iloc[i].tolist()

        tr_x.append(dp[0:col_count-1])
        tr_y.append(dp[col_count-1])

    return numpy.array(tr_x), numpy.array(tr_y)

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
        te_x.append(dp[0:col_count-1])
        te_y.append(dp[col_count-1])
    
    return numpy.array(te_x), numpy.array(te_y)

def part_c(tr_x, tr_y):
    gammas = [0.1, 0.5, 1, 5, 100]
    gdsvm = GDualSVM.GDualSVM()

    gamma_svis = {0.1:None, 0.5:None, 1:None, 5:None, 100:None}

    for g in gammas:
        gdsvm.create_classifier(tr_x, tr_y, 500/873, g)
        gamma_svis[g] = gdsvm.svis.copy()
    
    print("0.1 to 0.5 similarities:", len(numpy.intersect1d(gamma_svis[0.1], gamma_svis[0.5])))
    print("0.5 to 1 similarities:", len(numpy.intersect1d(gamma_svis[0.5], gamma_svis[1])))
    print("1 to 5 similarities:", len(numpy.intersect1d(gamma_svis[1], gamma_svis[5])))
    print("5 to 100 similarities:", len(numpy.intersect1d(gamma_svis[5], gamma_svis[100])))
        

def dual_test2(tr_x, tr_y, te_x, te_y):
    global C, GSTART
    gdsvm = GDualSVM.GDualSVM()

    for g0 in GSTART:
        print("===== Gamma Start:", g0, "======")
        
        for c in C:
            print("\t=== C:", c, "===")

            gdsvm.create_classifier(tr_x, tr_y, c, g0)

            print("\tGaussian Kernel")
            print("\t\tWeights:", gdsvm.weights)
            print("\t\tBias:", gdsvm.bstar)
            print("\t\t# of Support Vectors", len(gdsvm.svis))
            
            tr_pred = gdsvm.gaussian_predict(tr_x, tr_x, tr_y, g0)
            wrong = 0
            for pi in range(len(tr_pred)):
                if tr_pred[pi] != tr_y[pi]:
                    wrong += 1
            
            print("\t\tTraining Error:", str(wrong), "(", str(wrong / 872 * 100), "%)")

            te_pred = gdsvm.gaussian_predict(te_x, tr_x, tr_y, g0)
            wrong = 0
            for pi in range(len(te_pred)):
                if te_pred[pi] != te_y[pi]:
                    wrong += 1
            
            print("\t\tTesting Error:", str(wrong), "(", str(wrong / 500 * 100), "%)")
##

def dual_test1(tr_x, tr_y, te_x, te_y):
    global C
    dsvm1 = DualSVM.DualSVM()

    for c in C:
        print(str(c))
        dsvm1.create_classifier(tr_x, tr_y, c)
        print("\tWeights:", dsvm1.weights)
        print("\tBias:", dsvm1.bias)
        
        tr_pred = dsvm1.predict_all(tr_x)
        wrong = 0
        for pi in range(len(tr_pred)):
            if tr_pred[pi] != tr_y[pi]:
                wrong += 1
        
        print("\tTraining Error:", str(wrong), "(", str(wrong / 872 * 100), "%)")

        te_pred = dsvm1.predict_all(te_x)
        wrong = 0
        for pi in range(len(te_pred)):
            if te_pred[pi] != te_y[pi]:
                wrong += 1
            
        print("\tTesting Error:", str(wrong), "(", str(wrong / 500 * 100), "%)")
        print("=====")

def primal_test(tr_x, tr_y, te_x, te_y):
    global C, T
    psvm1 = PrimalSVM.PrimalSVM(util.schedule_one)
    psvm2 = PrimalSVM.PrimalSVM(util.schedule_two)

    for c in C:
        print("=== C:", c, "===")

        tr1_error_avg = 0
        te1_error_avg = 0
        tr2_error_avg = 0
        te2_error_avg = 0

        s1_weights = numpy.array([[0., 0., 0., 0., 0.]])
        s2_weights = numpy.array([[0., 0., 0., 0., 0.]])
        
        for i in range(10):
            psvm1.create_classifier(tr_x, tr_y, c, 0.1, 0.01)
            psvm2.create_classifier(tr_x, tr_y, c, 0.1, 0.01)

            # print("\tSchedule 1")
            # print("\t\tWeights:", psvm1.weight)

            s1_weights += psvm1.weight
            s2_weights += psvm2.weight

            # schedule 1
            tr_pred = psvm1.predict_all(tr_x)
            wrong = 0
            for pi in range(len(tr_pred)):
                if tr_pred[pi] != tr_y[pi]:
                    wrong += 1
            
            # print("\t\tTraining Error:", str(wrong), "(", str(wrong / 872 * 100), "%)")
            tr1_error_avg += wrong

            te_pred = psvm1.predict_all(te_x)
            wrong = 0
            for pi in range(len(te_pred)):
                if te_pred[pi] != te_y[pi]:
                    wrong += 1
            te1_error_avg += wrong
                
            # print("\t\tTesting Error:", str(wrong), "(", str(wrong / 500 * 100), "%)")

            # print("\tSchedule 2")
            # print("\t\tWeights:", psvm2.weight)

            # schedule 2
            tr_pred = psvm2.predict_all(tr_x)
            wrong = 0
            for pi in range(len(tr_pred)):
                if tr_pred[pi] != tr_y[pi]:
                    wrong += 1
            tr2_error_avg += wrong
            # print("\t\tTraining Error:", str(wrong), "(", str(wrong / 500 * 100), "%)")

            te_pred = psvm2.predict_all(te_x)
            wrong = 0
            for pi in range(len(te_pred)):
                if te_pred[pi] != te_y[pi]:
                    wrong += 1
            te2_error_avg += wrong
            # print("\t\tTesting Error:", str(wrong), "(", str(wrong / 500 * 100), "%)")
        
        print("\tAverage Weight Schedule 1:", (s1_weights / 10))
        tr1_error_avg /= 10
        print("\tAverage Training Error Schedule 1:", str(tr1_error_avg), "(", str(tr1_error_avg / 872 * 100), "%)")
        te1_error_avg /= 10
        print("\tAverage Training Error Schedule 1:", str(te1_error_avg), "(", str(te1_error_avg / 500 * 100), "%)")
        
        print("\tAverage Weight Schedule 2:", (s2_weights / 10))
        tr2_error_avg /= 10
        print("\tAverage Training Error Schedule 2:", str(tr2_error_avg), "(", str(tr2_error_avg / 872 * 100), "%)")
        te2_error_avg /= 10
        print("\tAverage Training Error Schedule 2:", str(te2_error_avg), "(", str(te2_error_avg / 500 * 100), "%)")
##

def main():

    tr_x, tr_y = parse_train()
    te_x, te_y = parse_test()

    print("VVVVVVVVVVVVVV PRIMAL TESTING VVVVVVVVVVVVVV")
    primal_test(tr_x, tr_y, te_x, te_y)
    print("^^^^^^^^^^^^^^ PRIMAL TESTING ^^^^^^^^^^^^^^")

    print("VVVVVVVVVVVVVV DUAL TESTING VVVVVVVVVVVVVV")
    dual_test1(tr_x, tr_y, te_x, te_y)
    dual_test2(tr_x, tr_y, te_x, te_y)
    part_c(tr_x, tr_y)
    print("^^^^^^^^^^^^^^ DUAL TESTING ^^^^^^^^^^^^^^")

    # FIN
    return 0

###

if __name__ == "__main__":
    main()