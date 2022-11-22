import numpy

def schedule_one(g0, a, t):
    return g0 / (1 + g0 / a * t)

def schedule_two(g0, a, t):
    return g0 / (1 + t)

def shuffle(Dx, shuffle_count):
    # Generate a random array of indices
    shuffled_indices = []

    row_count = Dx.shape[0]
    for _ in range(shuffle_count):
        shuffled = numpy.arange(row_count)
        numpy.random.default_rng().shuffle(shuffled)
        shuffled_indices.append(shuffled)

    return shuffled_indices
