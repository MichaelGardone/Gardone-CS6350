import numpy

class BatchGD:
    def __init__(self, threshold=1e-5, max_iterations=500, lr=1):
        self._threshold = threshold
        self._lr = lr
        self._max_iterations = max_iterations

        self._costHistory = {}
        self._costHistorySingleDescent = []

    def descent(self, x, y):
        # w should be of size |x[0]| because it has all of our features we care about
        w = numpy.zeros_like(x[0])
        lastCost = 9999
        
        for it in range(self._max_iterations):
            # print("====", it, "====")
            # print("curr w", w)
            # compute gradient
            gradient = numpy.zeros_like(w)
            
            # gradient[j] = -SUM((yi - wT . xi) * xij)
            for j in range(len(gradient)):
                for xi, yi in zip(x,y):
                    gradient[j] -= (yi - numpy.dot(w, xi)) * xi[j]
            
            # print("grad", gradient)

            # update w
            w = w - self._lr * gradient

            # print("new w", w)

            # cost of current w
            cost = 0.0
            for xi,yi in zip(x,y):
                cost += numpy.square(yi - numpy.dot(w, xi))
            # print("cost", cost)
            
            # Did we get a small enough difference that we can break out?
            diff_from_last = abs(cost - lastCost)
            lastCost = cost

            # preserve for output
            self._costHistorySingleDescent.append(cost)

            # if threshold met, break because we found our w!
            # also, make answ equal to something so we don't get locked in...
            if diff_from_last < self._threshold:
                break
        
        return w, it

    def descent_until_converge(self, x, y):
        curr_lr = self._lr
        answ = None

        while answ == None:
            lastCost = 9999
            # w should be of size |x[0]| because it has all of our features we care about
            w = numpy.zeros_like(x[0])

            self._costHistory[curr_lr] = []

            for it in range(self._max_iterations):
                # print("====", it, "====")
                # print("curr w", w)
                # compute gradient
                gradient = numpy.zeros_like(w)
                
                # gradient[j] = -SUM((yi - wT . xi) * xij)
                for j in range(len(gradient)):
                    for xi, yi in zip(x,y):
                        gradient[j] -= (yi - numpy.dot(w, xi)) * xi[j]
                
                # print("grad", gradient)

                # update w
                w = w - curr_lr * gradient

                # print("new w", w)

                # cost of current w
                cost = 0.0
                for xi,yi in zip(x,y):
                    cost += numpy.square(yi - numpy.dot(w, xi))
                # print("cost", cost)
                
                # Did we get a small enough difference that we can break out?
                diff_from_last = abs(cost - lastCost)
                lastCost = cost

                # preserve for output
                self._costHistory[curr_lr].append(cost)

                # if threshold met, break because we found our w!
                # also, make answ equal to something so we don't get locked in...
                if diff_from_last < self._threshold:
                    answ = (w, it + (len(self._costHistory) - 1) * self._max_iterations, curr_lr)
                    break
            
            curr_lr *= 0.5

        return answ
    
    def cost_lms(self, data, weight):
        sum_cost = 0.0
        
        for xi,yi in data:
            sum_cost += numpy.square(yi - numpy.dot(weight, xi))
        
        return 0.5 * sum_cost
    
    def get_cost_history(self):
        return self._costHistory

def predict(data, weights):
    pass
