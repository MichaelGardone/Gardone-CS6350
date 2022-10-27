import numpy

class VPerceptron():
    def __init__(self, shuffled_indices=None):
        self._ws = []   # all weights resulting from init/update_perceptron
        self._cs = []   # the total # of right predictions
        self._shuffled_indices = shuffled_indices

    def shuffle(self, Dx, shuffle_count):
        # Generate a random array of indices
        shuffled_indices = []

        row_count = Dx.shape[0]
        for _ in range(shuffle_count):
            shuffled = numpy.arange(row_count)
            numpy.random.default_rng().shuffle(shuffled)
            shuffled_indices.append(shuffled)

        return shuffled_indices

    def init_perceptron(self, Dx, Dy, r=0.1, T=10):
        # clean out ws and cs
        self._ws.clear()
        self._cs.clear()

        # Weight vector
        self._w = numpy.zeros_like(Dx[0])
        # index into cs
        m = -1

        shuffled_indices = None
        if self._shuffled_indices is not None:
            shuffled_indices = self._shuffled_indices
        else:
            shuffled_indices = self.shuffle(Dx, T)

        for epoch in range(T):
            for indx in shuffled_indices[epoch]:
                if Dy[indx] * numpy.dot(self._w, Dx[indx]) <= 0:
                    if m != -1:
                        self._ws.append(numpy.copy(self._w))
                    self._w += r * Dy[indx] * Dx[indx]
                    m += 1
                    self._cs.append(1)
                else:
                    self._cs[m] += 1
        
        # if we hit the end and we dont have the last w update in there (which we won't)
        #   add it
        self._ws.append(numpy.copy(self._w))

        ###

    def update_perceptron(self, Dx, Dy, r=0.1, T=10):
        shuffled_indices = self.shuffle(Dx, T)

        m = len(self._cs)

        for epoch in range(T):
            for indx in shuffled_indices[epoch]:
                if Dy[indx] * numpy.dot(self._w, Dx[indx]) <= 0:
                    self._ws.append(numpy.copy(self._w))
                    w += r * Dy[indx] * Dx[indx]
                    m += 1
                    self._cs.append(1)
                else:
                    self._cs[m] += 1
        
        ###
    
    def predict(self, Dx):
        predictions = []
        
        for x in Dx:
            s = 0
            for i in range(len(self._cs)):
                s += self._cs[i] * numpy.sign(numpy.dot(x, self._ws[i]))
            predictions.append(numpy.sign(s))

        return predictions
    
    def get_weights(self):
        res = []

        for i in range(len(self._cs)):
            res.append((self._ws[i], self._cs[i]))

        return res
