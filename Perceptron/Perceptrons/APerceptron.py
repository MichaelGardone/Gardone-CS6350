import numpy

class APerceptron():
    def __init__(self, shuffled_indices=None):
        self._w = None
        self._a = None
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
        # Weight vector
        self._w = numpy.zeros_like(Dx[0])
        self._a = numpy.zeros_like(self._w)

        shuffled_indices = None
        if self._shuffled_indices is not None:
            shuffled_indices = self._shuffled_indices
        else:
            shuffled_indices = self.shuffle(Dx, T)

        for epoch in range(T):
            for indx in shuffled_indices[epoch]:
                if Dy[indx] * numpy.dot(self._w, Dx[indx]) <= 0:
                    self._w += r * Dy[indx] * Dx[indx]
                self._a += self._w
        
        ###
    
    def update_perceptron(self, Dx, Dy, r=0.1, T=10):
        shuffled_indices = self.shuffle(Dx, T)

        for epoch in range(T):
            for indx in shuffled_indices[epoch]:
                if Dy[indx] * numpy.dot(self._w, Dx[indx]) <= 0:
                    self._w += r * Dy[indx] * Dx[indx]
            self._a += self._w
        ###
    
    def predict(self, Dx):
        predictions = []
        
        for x in Dx:
            predictions.append(numpy.sign(numpy.dot(x, self._a)))

        return predictions
    
    def get_weights(self):
        return self._w
