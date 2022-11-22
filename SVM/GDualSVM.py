import numpy, copy
from scipy.optimize import minimize,Bounds

import util

class GDualSVM():
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.update_count = 0

        self._THRESHOLD = 1e-10

    def create_classifier(self, x, y, C, g):
        # Bias needs to be recovered on its own, can't be sneaky :(
        # self.weight = numpy.zeros(x[0].shape[0] - 1)
        alphas0 = numpy.zeros(x.shape[0]) # number of alphas = number of examples

        CONSTRAINTS = [
            # sum constraint
            { 'type':'eq', 'fun': lambda a: numpy.dot(a,y) },
        ]
        BOUNDS = [(0,C)] * x.shape[0]

        self.alphas = minimize(self._gaussian_objective_func, alphas0, args=(x, y, g), method='SLSQP', constraints=CONSTRAINTS, bounds=BOUNDS).x

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
        self.bias = y - numpy.sum((self.alphas * y) * self._gaussian_kernel(x, x, g))
        self.bstar = numpy.mean(self.bias)
    
    def _gaussian_objective_func(self, alphas, x, y, g):
        # args comes from args in minimized
        kernel = self._gaussian_kernel(x, x, g)

        res = y.T * kernel * y
        res = alphas.T.dot(res * alphas)

        return 0.5 * res - numpy.sum(alphas)
    
    def _gaussian_kernel(self, xi, xj, g):
        return numpy.exp(-numpy.sum(numpy.square(xi - xj), axis=1) / g)
    
    def _single_gaussian_kernel(self, xi, xj, g):
        return numpy.exp(-numpy.square(numpy.linalg.norm(xi - xj)) / g)

    def gaussian_predict(self, x, tr_x, tr_y, gamma):
        if len(x) > len(tr_x):
            lx = x[:len(tr_x)]
            ltrx = tr_x
            ltry = tr_y
            lalphas = self.alphas
            lbias = self.bias
        else:
            lx = x
            ltrx = tr_x[:len(x)]
            ltry = tr_y[:len(x)]
            lalphas = self.alphas[:len(x)]
            lbias = self.bias[:len(x)]

        k = self._gaussian_kernel(ltrx, lx, gamma)

        return numpy.sign((lalphas * ltry) * k + lbias)
