import numpy, copy, util

class Sigmoid():
    def __init__(self) -> None:
        pass
    ##

    def derivative(self, prior_deriv, prior_layer, next_layer):
        return prior_deriv * next_layer * prior_layer * (1 - prior_layer)
    ##
    
    def calculate(self, x):
        return 1 / (1 + numpy.exp(-x))
    ##

    def weight_derivatives(self, wderiv, prior_derivs, prev_layer, next_layer):
        lwderiv = len(wderiv)

        for i in range(lwderiv):
            for j in range(1, len(wderiv[i]) + 1):
                wderiv[i][j-1] = self.derivative(prior_derivs[j], prev_layer[j], next_layer[i])
        
        return wderiv
    ##

    def node_derivatives0(self, weights, prior_derivs, curr_layer):
        node_derivs = numpy.zeros(curr_layer.shape)

        for i in range(len(node_derivs)):
            accu = 0

            for j in range(len(weights[i])):
                accu += weights[i][j] * prior_derivs[j]
            
            node_derivs[i] = accu
        
        return node_derivs
    ##

    def node_derivatives(self, weights, prior_derivs, curr_layer):
        node_derivs = numpy.zeros(curr_layer.shape)

        for i in range(len(node_derivs)):
            accu = 0

            for j in range(len(weights[i])):
                accu += weights[i][j] * prior_derivs[j + 1]
            
            node_derivs[i] = accu
        
        return node_derivs
    ##

    def activation(self, x, weights, width):
        layer_results = numpy.zeros(width)
        layer_results[0] = 1

        for i in range(width - 1):
            accu = 0
            
            for j in range(len(x)):
                accu += x[j] * weights[j][i]
            
            layer_results[i + 1] = self.calculate(accu)
            
        return layer_results
    ##
###

class Linear():
    def __init__(self) -> None:
        pass
    ##

    def derivative(self, prior_deriv, prior_layer, next_layer):
        return prior_deriv[0] * next_layer
    ##

    def activation(self, x, weights, width):
        curr_results = numpy.zeros(width)

        for i in range(width):
            for j in range(len(x)):
                curr_results[i] += x[j] * weights[j][i]
            
        return curr_results
    ##

    def node_derivatives(self, weights, prior_derivs, curr_layer):
        node_derivs = numpy.zeros(curr_layer.shape)

        for i in range(len(node_derivs)):
            accu = 0

            for j in range(len(weights[i])):
                accu += weights[i][j] * prior_derivs[j + 1]
            
            node_derivs[i] = accu
        
        return node_derivs
    ##

    def node_derivatives0(self, weights, prior_derivs, curr_layer):
        node_derivs = numpy.zeros(curr_layer.shape)

        for i in range(len(node_derivs)):
            accu = 0

            for j in range(len(weights[i])):
                accu += weights[i][j] * prior_derivs[j]
            
            node_derivs[i] = accu
        
        return node_derivs
    ##

    def weight_derivatives(self, wderiv, prior_derivs, prev_layer, next_layer, actfunc):
        for i in range(len(wderiv)):
            for j in range(len(wderiv[i])):
                wderiv[i][j] = actfunc.derivative(prior_derivs[j], prev_layer[j], next_layer[i])
        
        return wderiv
    ##
###

class NeuralNetwork():
    def __init__(self, layers, weights, layer_widths):
        self._layers = layers
        self._weights = weights
        self._layer_widths = layer_widths

        # results from a single iteration of forward propagation to be used in backwards propogation
        self._results = None
    ##

    def predict_all(self, xs):
        predictions = []

        for x in xs:
            predictions.append(self.predict(x))

        return predictions
    ##

    def predict(self, x):
        """
        A wrapper around the forward function, because that's all prediction is in NNs.
        Just makes it cleaner, I guess?
        """
        return numpy.sign(self._forward(x))
    ##

    def train(self, data, classifications, d, lr, T=100):
        shuffled_indices = util.shuffle(data, T)

        for t in range(T):
            lri = self._gamma_schedule(lr, d, t)

            for i in shuffled_indices[t]:
                x = data[i]
                y = classifications[i]

                res = self._forward(x)
                wderivs = self._backward(res - y)

                # update weights
                self._update_weights(lri, wderivs)
    ##

    def _update_weights(self, lr, derivatives):
        for i in range(len(derivatives)):
            for j in range(len(derivatives[i])):
                for k in range(len(derivatives[i][j])):
                    self._weights[i][j][k] -= lr * derivatives[i][j][k]
    ##

    def _gamma_schedule(self, g0, d, t):
        return g0 / (1 + g0 / d * t)
    ##

    def _forward(self, x):
        self._results = numpy.array([numpy.zeros(w) for w in self._layer_widths], dtype=object)
        
        # baseline is always the example
        self._results[0] = x

        for i in range(1, len(self._layer_widths)):
            self._results[i] = self._layers[i-1].activation(self._results[i-1], self._weights[i-1], self._layer_widths[i])

        return self._results[-1]
    ##

    def _backward(self, loss, verbose=False):
        pn_derivs = [loss]
        wderivs = copy.deepcopy(self._weights)

        start = len(self._weights) - 1

        # initial weight --> because it's linear, and always will be, we can use a special derivative function for the nodes
        wderivs[start] = self._layers[start].weight_derivatives(wderivs[start], pn_derivs, self._results[start+1], self._results[start], self._layers[start])
        pn_derivs = self._layers[start].node_derivatives0(self._weights[start], pn_derivs, self._results[start])
        if verbose:
            print(pn_derivs)

        if verbose:
            for li in range(start - 1, 0, -1):
                wderivs[li] = self._layers[li].weight_derivatives(wderivs[li], pn_derivs, self._results[li+1], self._results[li])
                pn_derivs = self._layers[start].node_derivatives(self._weights[li], pn_derivs, self._results[li])
                print(pn_derivs)
        else:
            for li in range(start - 1, 0, -1):
                wderivs[li] = self._layers[li].weight_derivatives(wderivs[li], pn_derivs, self._results[li+1], self._results[li])
                pn_derivs = self._layers[start].node_derivatives(self._weights[li], pn_derivs, self._results[li])
                
        # final weight derivative from the w^1 weights
        wderivs[0] = self._layers[0].weight_derivatives(wderivs[0], pn_derivs, self._results[1], self._results[0])

        # don't do pn_derivs, the derivative of input layer nodes doesn't make sense

        return wderivs
    ##
###