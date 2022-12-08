import numpy, copy

def zero_init(width, nwidth):
    weights = []

    for _ in range(width):
        weights.append([0] * ((nwidth - 1) if nwidth - 1 > 0 else 1))

    return weights
##

def rand_init(width, nwidth):
    weights = []

    for _ in range(width):
        t = numpy.random.normal(0, 0.1, nwidth - 1 if nwidth - 1 > 0 else 1).tolist()
        weights.append(t)

    return weights
##

def backprop_check_l12(width, nwidth):
    return numpy.array([[-1,-2,-3],[1,2,3]])
##

def backprop_check_l3(width, nwidth):
    return numpy.array([[-1,2,-1.5]])
##

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
