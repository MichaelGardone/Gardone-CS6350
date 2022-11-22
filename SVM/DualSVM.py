import numpy, copy
from scipy.optimize import minimize,Bounds

import util

class DualSVM():
    def __init__(self, use_gaussian=False, bias_recovery=None):
        self.use_gaussian = use_gaussian
        self.bias_recovery = bias_recovery

        self.weights = None
        self.bias = 0
        self.update_count = 0

        self._THRESHOLD = 1e-10

    def create_classifier(self, x, y, C):
        """
            :param: x -> training data points
            :param: y -> training data labels
            :param: C -> the constant for the hinge loss function
            :param: g -> gamma/learning rate
        """

        # Bias needs to be recovered on its own, can't be sneaky :(
        # self.weight = numpy.zeros(x[0].shape[0] - 1)
        alphas0 = numpy.zeros(x.shape[0]) # number of alphas = number of examples

        CONSTRAINTS = [
            # sum constraint
            { 'type':'eq', 'fun': lambda a: numpy.dot(a,y) },
        ]
        BOUNDS = [(0,C)] * x.shape[0]

        minimized = minimize(self._objective_func, alphas0, args=(x, y), method='SLSQP', constraints=CONSTRAINTS, bounds=BOUNDS)
        
        # re-alias
        self.alphas = minimized.x

        # Support vectors -- everything on margin or inside
        self.svis = []
        for i, a in enumerate(self.alphas):
            if a > self._THRESHOLD:
                self.svis.append(i)

        # Weight recovery
        self.weights = numpy.zeros_like(x[0])
        for i in range(x.shape[0]):
            self.weights += self.alphas[i] * y[i] * x[i]
        
        # bias recovery -- need to use kernel that was used in minimize() (or something adjacent to it)
        self.bias = 0
        for i in self.svis:
            self.bias += y[i] - self._single_linear_kernel(self.weights, x[i])
        
        self.bias = self.bias / len(self.svis)

    def _linear_kernel(self, xi, xj):
        return xi @ xj.T
    
    def _single_linear_kernel(self, xi, xj):
        return numpy.dot(xi, xj)
    
    def _objective_func(self, alphas, x, y):
        # args comes from args in minimized
        y = y * numpy.ones((len(y), len(y)))
        a = alphas * numpy.ones((len(alphas), len(alphas)))
        kernel = self._linear_kernel(x, x)

        yakern = (y * y.T) * (a * a.T) * kernel

        return 0.5 * numpy.sum(yakern) - numpy.sum(a)
    
    def predict_all(self, data, g=0):
        predictions = []
        
        for xi in data:
            predictions.append(self.predict(xi, g))
        
        return predictions

    def predict(self, x, g):
        # I already store the weights + bias, so meh
        return numpy.sign(numpy.dot(self.weights, x) + self.bias)
