import numpy

def gradient_descent(x, y, lr=1, threshold=1e-6,max_iterations=1000):
    curr_lr = lr
    avg_cost_history = []
    converge = False
    weight = None
    final_costs = []
    iters = 0

    while converge == False:
        w = numpy.zeros((1, x.shape[1]))
        costs = []
        
        for it in range(max_iterations):
            # compute gradient
            # compute w^T * x
            xwt = x * numpy.transpose(w)
            # error = y - w^T x
            error = y - xwt
            # -sum((error) * xij)
            gradient = -numpy.transpose(x) * error

            # update w
            next_w = w - curr_lr * numpy.transpose(gradient)

            # cost of new w (wp1)
            costs.append(lms_cost(x, y, next_w))
            
            # if threshold met, break because we found our w!
            # also, make answ equal to something so we don't get locked in...
            if numpy.linalg.norm(next_w - w) <= threshold:
            # if cost - last_cost <= threshold:
                converge = True
                weight = next_w
                final_costs = costs
                iters = it
                break
            
            w = next_w

        # nan / inf check
        avg_cost = 0
        non_inf_values = []
        for i in range(len(costs)):
            if numpy.isfinite(costs[i]) == False:
                break
            non_inf_values.append(costs[i])
        
        for v in non_inf_values:
            avg_cost += v / len(non_inf_values)

        avg_cost_history.append((curr_lr, avg_cost, -1 if numpy.isfinite(costs[i]) else i))
        
        curr_lr *= 0.5

    return weight, avg_cost_history, final_costs, iters

def lms_cost(x, y, w):
    cost = 0.0
    for i in range(len(x)):
        cost += numpy.square(y[i] - x[i] * numpy.transpose(w))
    return cost * 0.5

