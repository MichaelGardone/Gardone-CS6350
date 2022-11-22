import numpy, copy

import util

def run_primal_svm(x, y, g0, a):
    pass

class PrimalSVM():
    """
    Primal SVM using Stochastic Gradient Descent.
    """

    def __init__(self, lr_func=util.schedule_two):
        self.lr_function = lr_func

        self.weight = None
        self.update_count = 0

    def create_classifier(self, x, y, C, g, a, T=100):
        """
            :param: x -> training data points
            :param: y -> training data labels
            :param: C -> the constant for the hinge loss function
            :param: g -> gamma/learning rate
            :param: a -> used in schedule function 1, ignored in 2
            :param: T -> Number of epochs
        """
        lx = numpy.insert(x, x.shape[1], [1]*x.shape[0], axis=1)
        # Randomly shuffle data --> could be done anywhere but took this function from
        #                           my Perceptron implementation
        shuffled_indices = util.shuffle(x, T)
        N = x.shape[0]

        self.weight = numpy.array([[0., 0., 0., 0., 0.]])

        for t in range(T):
            gamma = self.lr_function(g, a, t)

            # for each index in the [shuffled] data set
            for i in shuffled_indices[t]:
                if numpy.dot(lx[i], numpy.transpose(self.weight)) * y[i] <= 1:
                    self.weight = (1-gamma) * self.weight + gamma * C * N * y[i] * lx[i]
                else:
                    self.weight = (1-gamma) * self.weight[:self.weight.size-1]

    def create_verbose_classifier(self, x, y, C):
        """
            :param: x -> training data points
            :param: y -> training data labels
            :param: C -> the constant for the hinge loss function
            :param: T -> Number of epochs
        """

        # Randomly shuffle data --> could be done anywhere but took this function from
        #                           my Perceptron implementation
        shuffled_indices = util.shuffle(x, 3)
        N = x.shape[0]

        self.weight = numpy.array([[0., 0., 0., 0.]])
        # self.weight_history = []

        # self.weight_history.append(self.weight)
        # lr = self.lr_function(g, a, t)
        lrs = [0.01, 0.005, 0.0025]

        for lri in range(len(lrs)):
            gamma = lrs[lri]
            print(str(lri))
            # for each index in the [shuffled] data set
            for i in shuffled_indices[lri]:

                if numpy.dot(x[i], numpy.transpose(self.weight)) * y[i] <= 1:
                    self.weight = (1-gamma) * self.weight + gamma * C * N * y[i] * x[i]
                else:
                    self.weight = (1-gamma) * self.weight[:self.weight.size-1]

                print("\t", i, ":", self.weight)
            print("=========")

    def predict_all(self, x):
        predictions = []

        ld = numpy.insert(x, x.shape[1], [1]*x.shape[0], axis=1)

        for xi in ld:
            predictions.append(self.single_predict(xi))
        
        return predictions

    def single_predict(self, x):
        return numpy.sign(numpy.dot(self.weight, x))
