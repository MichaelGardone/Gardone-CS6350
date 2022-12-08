import numpy

def shuffle(Dx, shuffle_count):
    # Generate a random array of indices
    shuffled_indices = []

    row_count = Dx.shape[0]
    for _ in range(shuffle_count):
        shuffled = numpy.arange(row_count)
        numpy.random.default_rng().shuffle(shuffled)
        shuffled_indices.append(shuffled)

    return shuffled_indices
##

class LogisticRegression():
    def __init__(self, nweights, method="ml") -> None:
        if method in ["ml", "mle", "map_e"]:
            self.loss = self._mle_loss
            self._gradient_func = self._mle_gradient
        elif method in ["map", "mape", "map_e"]:
            self.loss = self._map_loss
            self._gradient_func = self._map_gradient
        else:
            raise Exception("Unknown method!")
        
        self._weights = numpy.zeros(nweights)
    ##
    
    def train(self, x, y, var=0, d=1, lr=1, T=100):
        m = len(x)
        shuffled = shuffle(x, T)

        for t in range(T):
            lri = self._learning_schedule(t, lr, d)
            for i in shuffled[t]:
                xi = x[i]
                yi = y[i]
                
                grad = self._gradient_func(xi, yi, m, var)
                self._weights = self._weights - lri * grad
            ##
        ##
    ##

    def train_verbose(self, x, y, var=0, d=1, lr=1, T=100):
        m = len(x)
        shuffled = shuffle(x, T)
        print(shuffled)

        for t in range(T):
            lri = self._learning_schedule(t, lr, d)
            for i in shuffled[t]:
                xi = x[i]
                yi = y[i]
                
                grad = self._gradient_func(xi, yi, m, var)
                print(grad)
                self._weights = self._weights - lri * grad
            ##
        ##
    ##

    def _learning_schedule(self, t, g0, d):
        return g0 / (1 + g0 / d * t)
    ##

    def _mle_loss(self, x, y, _):
        return numpy.log(1 + numpy.exp(-y * numpy.dot(self._weights, x)))
    ##

    def _mle_gradient(self, x, y, m, _):
        ywx = y * numpy.dot(self._weights, x)
        log_deriv = 1 / (1 + numpy.exp(ywx))
        return -m * log_deriv * y * x
    ##

    def _map_loss(self, x, y, var):
        return numpy.log(1 + numpy.exp(-y * numpy.dot(self._weights, x))) + var * numpy.dot(self._weights, self._weights)
    ##

    def _map_gradient(self, x, y, m, var):
        ywx = y * numpy.dot(self._weights, x)
        log_deriv = 1 + numpy.exp(ywx)
        return -m * x * y / log_deriv + self._weights / var
    ##

    def predict_all(self, xs):
        predictions = []

        for x in xs:
            predictions.append(self.predict(x))
        
        return predictions
    ##

    def predict(self, x):
        return -1 if numpy.dot(self._weights, x) <= 0 else 1
    ##
###
