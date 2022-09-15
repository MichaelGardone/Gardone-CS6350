from cmath import exp
import math

class InformationGain:
    def gain(samples, attrib):
        pass

    def expected_measure(sample):
        pass

    def label_measure(sample):
        pass
    
    def measure(sample):
        pass

class EntropyGain(InformationGain):
    def gain(samples, attrib):
        pass

    def measure(sample):
        pass

def entropy(sv):
    '''
    The equivalent of H({p1, p2,...,pk}) = -SUM(k, i=1)pi * log2(pi) .\n
    [Arg] sv -> sv is the sub-sample of examples. Should only be a list of length 2 Something else, and the label.
    '''
    sample_count = len(sv)

    # if we have 0 or 1 sample, then there is no entropy here
    if sample_count < 2:
        return 0

    # Feature Label Count
    fl_count = {}

    for s in sv:
        if s[1] not in fl_count:
            fl_count[s[1]] = 0
        fl_count[s[1]] += 1
    
    entropy = 0
    total = sum(fl_count.values())
    for val in fl_count.values():
        prob = val / total
        entropy += -prob * math.log2(prob)

    return entropy

def label_entropy(sv_label):
    '''
    The equivalent of H({p1, p2,...,pk}) = -SUM(k, i=1)pi * log2(pi) .\n
    [Arg] sv -> sv is the sub-sample of examples. Should only be a list of length 2 Something else, and the label.
    '''
    sample_count = len(sv_label)

    # if we have 0 or 1 sample, then there is no entropy here
    if sample_count < 2:
        return 0

    # Feature Label Count
    fl_count = {}

    for s in sv_label:
        if s not in fl_count:
            fl_count[s] = 0
        fl_count[s] += 1
    
    entropy = 0
    total = sum(fl_count.values())
    for val in fl_count.values():
        prob = val / total
        entropy += -prob * math.log2(prob)

    return entropy

def expected_entropy(sv, attrib, values, label):
    count = len(sv)

    expected_entropy = 0
    for v in values:
        ssv = []
        for s in sv:
            if (s[attrib] == v):
                ssv.append([s[attrib], s[label]])
        calc_entropy = entropy(ssv)
        expected_entropy += len(ssv) / count * calc_entropy

    return expected_entropy

def information_gain(sv, feature, feature_values, label):
    # Total Entropy
    ssv = []
    for s in sv:
        ssv.append(s[label])
    le = label_entropy(ssv)

    # Expected Entropy
    ee = expected_entropy(sv, feature, feature_values, label)

    return le - ee

def gini(sv):
    pass

def majority_error(sv):
    pass