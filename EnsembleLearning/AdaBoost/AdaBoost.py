from DecisionTree import id3, gain
import numpy, copy

class AdaBoost:
    def __init__(self, label, sample_count, gain=gain.EntropyGain(), debug=False) -> None:
        self._label = label
        self._gain = gain
        self._hypotheses = [] # entries take the form of a tuple: (EID3 tree, vote_weight)
        self._sample_count = sample_count

        # Debug Properties
        self._debug  = debug # if true, record each iter of _D and error
        self._prevD  = [] # a list of lists with the full list being equal to T and each sublist being equal to size len(self._D) or sample_count
        self._errors = [] # should always equal len(self._hypotheses)

    def boost(self, examples, attributes, T=250):
        '''
            Run single_boost T number of times.
        '''
        actual_labels = examples[self._label].tolist()
        
        # for t = 1, 2, ..., T
        for t in range(T):
            self.single_boost(examples, attributes, actual_labels)

    def single_boost(self, examples, attributes, actual_labels):
        '''
            Commit 1 additional boost.
        '''
        self.weights = examples["weight"].tolist()

        # Copy to record for graphs potentially
        # if self._debug:
        #     self._prevD.append(copy.deepcopy(self.weights))

        # Find a classifier h_t whose weighted classification error is better than chance
        # >>> Just get the label with the best gain
        # !-> Don't send weights through here, let ID3 associate the weights on its own otherwise confusion
        root = id3.EID3(self._label, self._gain).generate_stump(examples, attributes)
        
        # What does the tree predict?
        predictions = self.classify_single_h(root, examples)

        # What is the weighted classification error based on this tree?
        error = self._compute_error(predictions, actual_labels)

        if self._debug:
            self._errors.append(error)

        # Compute its vote
        alpha = self._compute_vote(error)
        
        # add it to the hypotheses list along with its weight (vote)
        self._hypotheses.append((root, alpha))

        # Update the values of the weights for the training examples
        self._update_weights(alpha, actual_labels, predictions)

        # Update weights in panda
        examples["weight"] = self.weights

    def _compute_vote(self, error):
        return 0.5 * numpy.log((1 - error) / error)
    
    def _compute_error(self, y, x):
        # e = 1/2 - 1/2 (SUM(D_t(i) * yi * ht(xi))
        # wsum = 0

        # # (SUM(D_t(i) * yi * ht(xi))
        # for i in range(self._sample_count):
        #     wsum += self.weights[i] * y[i] * x[i]

        # return 0.5 - 0.5 * wsum
        error = 0

        for i in range(self._sample_count):
            if y[i] != x[i]:
                error += self.weights[i]

        return error
    
    def _update_weights(self, alpha, actual_labels, predictions):
        # print(self.weights)

        # normalization constant -- make sure SUM(D_{t+1}) == 1
        Z = 0

        # D_t(i) * exp(-alpha * yi * ht(x_i) )
        for i in range(self._sample_count):
            self.weights[i] = self.weights[i] * numpy.exp(-alpha * actual_labels[i] * predictions[i])
            Z += self.weights[i]

        # print(Z)

        for i in range(self._sample_count):
            self.weights[i] = self.weights[i] / Z

        # Check to make sure SUM(D) = 1
        # print(self.weights)
        # print(sum(self.weights))
    
    def classify_single_h(self, hypothesis, data):
        pred_results = []
        
        for i in range(self._sample_count):
            pred_results.append(int(hypothesis.predict(data.iloc[i])))
        
        return pred_results

    def classify(self, inst):
        wsum = 0

        for h, alpha in self._hypotheses:
            wsum += alpha * int(h.predict(inst))
        
        return numpy.sign(wsum)
    
    def test_hypothesis_at(self, data, index=0):
        return int(self._hypotheses[index][0].predict(data))
    
    def verbose_classify(self, inst):
        wsum = 0

        for i in range(len(self._hypotheses)):
            res = int(self._hypotheses[i][0].predict(inst))
            print("h (", str(self._hypotheses[i][1]), ")", i, "says", str(res))
            wsum += self._hypotheses[i][1] * res
        
        return numpy.sign(wsum)
    
    def print_hypotheses(self):
        for tree, alpha in self._hypotheses:
            print("====")
            print("VOTE WEIGHT:", alpha)
            tree.print_tree()
            print("====")

    def reset(self, gain=None):
        self._t = -1
        self._hypothesis.clear()
        if gain is not None:
            self.gain = gain
