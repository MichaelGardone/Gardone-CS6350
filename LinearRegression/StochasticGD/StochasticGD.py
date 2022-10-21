import numpy

def stch_gradient_descent(x, y, lr=1, threshold=1e-6, max_iterations=1000):
    curr_lr = lr
    avg_cost_history = []
    weight = None
    converge = False
    final_costs = []
    data_count = x.shape[0]
    feature_count = x.shape[1]
    iters = 0
    prev_cost = 9999

    while converge == False:
        w = numpy.zeros((1, x.shape[1]))
        costs = []

        for it in range(max_iterations):
            rand = numpy.random.randint(data_count)

            # our universe for now
            xi = x[rand]
            yi = y[rand]

            # gradient computation
            xwt = xi * numpy.transpose(w)
            error = yi - xwt

            for j in range(feature_count):
                w[0, j] = w[0, j] + curr_lr * error * xi[0, j]

            cost = lms_cost(x, y, w)
            costs.append(cost)

            if numpy.abs(cost - prev_cost) <= threshold:
                converge = True
                weight = w
                final_costs = costs
                iters = it
                break

            prev_cost = cost
        
        avg_cost = 0
        non_inf_values = []
        hits_inf = False
        for i in range(len(costs)):
            if numpy.isfinite(costs[i]) == False:
                hits_inf = True
                break
            non_inf_values.append(costs[i])
        
        for v in non_inf_values:
            avg_cost += v / len(non_inf_values)

        avg_cost_history.append((curr_lr, avg_cost, -1 if hits_inf == False else i))
        
        curr_lr *= 0.5

    return weight, avg_cost_history, final_costs, iters

def lms_cost(x,y,w):
    cost = 0.0
    for i in range(len(x)):
        cost += numpy.square(y[i] - x[i] * numpy.transpose(w))
    return cost * 0.5