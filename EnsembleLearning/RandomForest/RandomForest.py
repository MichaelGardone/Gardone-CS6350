from DecisionTree import id3, gain
import numpy, copy, random

class RandomForest:
    def __init__(self, label, sub_sample_count, sub_attrib_count, gain=gain.EntropyGain(), debug=False) -> None:
        self._label = label
        self._gain = gain
        self._hypotheses = [] # all hypotheses (trees) generated from ID3
        self._sample_count = sub_sample_count # the number of examples to pull from the training data
        self._attrib_count = sub_attrib_count # the number of attributes to go into G from A

        # Debug Properties
        self._debug = debug # if true, record each iter of _D and error

    def random_forest(self, examples, attributes, T=250):
        '''
            Run single_boost T number of times.
        '''
        
        for t in range(T):
            self.single_tree(examples, attributes)

    def single_tree(self, examples, attributes):
        '''
            Commit 1 additional tree to the random forest.
        '''

        # Get X number of rows, determined before entering tree creation
        sub_samples = examples.sample(n=self._sample_count)
        
        # Get G from A
        sub_attribs = {}
        a = list(attributes)
        while len(sub_attribs) < self._attrib_count:
            choice = random.choice(a)
            if choice not in sub_attribs:
                sub_attribs[choice] = attributes[choice]

        # Find a classifier h_t whose weighted classification error is better than chance
        # >>> Just get the label with the best gain
        root = id3.EID3(self._label, self._gain).generate_tree(sub_samples, sub_attribs)
        
        # add it to the hypotheses list along with its weight (vote)
        self._hypotheses.append(root)

    def classify(self, inst):
        vote = 0

        for h in self._hypotheses:
            vote += int(h.predict(inst)) # will return either -1 or 1, no average needed
        
        # if > 0, we classify as 1
        # if < 0, we classify as -1
        return numpy.sign(vote)
    
    def verbose_classify(self, inst):
        vote = 0

        for i in range(len(self._hypotheses)):
            res = int(self._hypotheses[i].predict(inst))
            print("h (", str(self._hypotheses[i].label), ")", i, "says", str(res))
            vote += res
        
        return numpy.sign(vote)
    
    def print_hypotheses(self):
        for tree in self._hypotheses:
            print("====")
            tree.print_tree()
            print("====")

    def reset(self, gain=None):
        self._t = -1
        self._hypothesis.clear()
        if gain is not None:
            self.gain = gain
