import numpy

class StochasticGD:
    def __init__(self, threshold=1e-5, max_iterations=500, lr=1):
        self._threshold = threshold
        self._lr = lr
        self._max_iterations = max_iterations

        self._weightHistory = {}
        self._costHistory = {}

    def descent(self, x, dimX, y):
        curr_lr = self._lr
        answ = None
        
        #while dfl > self._threshold and itera < self._max_iterations:
        while answ == None:
            # w should be of size |ex| because it has all of our features we care about
            ws = numpy.zeros([self._max_iterations + 1, dimX])
            ws = numpy.asmatrix(ws)
            # curr_lr should be added, in the for loop we will get out answers
            self._weightHistory[curr_lr] = ws
            self._costHistory[curr_lr] = []

            for i in range(self._max_iterations):
                # Get out the current matrix
                # At i = 0, that's just the zero vector
                w = ws[i]

                # compute lms cost, store for history
                self._costHistory[curr_lr].append(self.cost_lms(x, y, w))

                # compute gradient
                # get the tranpose of the weight vector
                wT = numpy.transpose(w)
                # error = y - W^TX
                error = y - x *  wT
                # gradient = -SUM(error * xij)
                gradient = -(numpy.transpose(x) * error)

                # gradient update
                # w = w - curr_lr * gradient (tranposed! -- don't forget that or numpy isn't happy...)
                ws[i + 1] = w - curr_lr * numpy.transpose(gradient)

                # Did we get a small enough difference that we can break out?
                diff_from_last = numpy.linalg.norm(ws[i+1] - w)

                # if threshold met, break because we found our w!
                # also, make answ equal to something so we don't get locked in...
                if diff_from_last < self._threshold:
                    answ = (ws[i+1], curr_lr)
                    break
            
            curr_lr *= 0.5
            
        return answ
    
    def cost_lms(self, data, label, weights):
        sum_cost = 0
        examples = int(data.size / 7)
        
        for exi in range(examples):
            sum_cost += numpy.square(label[exi] - data[exi] * numpy.transpose(weights))
        
        return 0.5 * sum_cost

    def get_weight_history(self):
        return self._weightHistory
    
    def get_cost_history(self):
        return self._costHistory
