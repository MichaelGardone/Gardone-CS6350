from cmath import exp
import math

class InformationGain:
    def gain(self, samples, feature, feature_values, label):
        pass

    def expected_measure(self, samples):
        pass

    def label_measure(self, label_samples):
        pass
    
    def measure(self, samples):
        pass

class EntropyGain(InformationGain):
    def gain(self, samples, feature, feature_values, label):
        # Total Entropy
        ssv = []
        for s in samples:
            ssv.append(s[label])
        le = self.label_measure(ssv)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label)

        return le - ee

    def measure(self, samples):
        '''
        The equivalent of H({p1, p2,...,pk}) = -SUM(k, i=1)pi * log2(pi) .\n
        [Arg] sv -> sv is the sub-sample of examples. Should only be a list of length 2 Something else, and the label.
        '''
        sample_count = len(samples)

        # if we have 0 or 1 sample, then there is no entropy here
        if sample_count < 2:
            return 0

        # Feature Label Count
        fl_count = {}

        for s in samples:
            if s[1] not in fl_count:
                fl_count[s[1]] = 0
            fl_count[s[1]] += 1
        
        entropy = 0
        total = sum(fl_count.values())
        for val in fl_count.values():
            prob = val / total
            entropy += -prob * math.log2(prob)

        return entropy
    
    def label_measure(self, label_samples):
        sample_count = len(label_samples)

        # if we have 0 or 1 sample, then there is no entropy here
        if sample_count < 2:
            return 0

        # Feature Label Count
        fl_count = {}

        for s in label_samples:
            if s not in fl_count:
                fl_count[s] = 0
            fl_count[s] += 1
        
        entropy = 0
        total = sum(fl_count.values())
        for val in fl_count.values():
            prob = val / total
            entropy += -prob * math.log2(prob)

        return entropy
    
    def expected_measure(self, samples, attrib, values, label):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            ssv = []
            for s in samples:
                if (s[attrib] == v):
                    ssv.append([s[attrib], s[label]])
            calc_entropy = self.measure(ssv)
            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy

class MajorityError(InformationGain):
    def gain(self, samples, feature, feature_values, label):
        # Total Entropy
        ssv = []
        for s in samples:
            ssv.append(s[label])
        le = self.label_measure(ssv)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label)

        return le - ee

    def measure(self, samples):
        count = len(samples)
        label_counts = {}

        if count == 0:
            return 0

        for s in samples:
            if s not in label_counts:
                label_counts[s] = 0
            label_counts[s] += 1

        res = label_counts[min(label_counts, key=label_counts.get)] / count
   
        return res if res < 1.0 else 0
    
    def label_measure(self, label_samples):
        count = len(label_samples)
        label_counts = {}

        for s in label_samples:
            if s not in label_counts:
                label_counts[s] = 0
            label_counts[s] += 1

        res = label_counts[min(label_counts, key=label_counts.get)] / count
   
        return res if res < 1.0 else 0
    
    def expected_measure(self, samples, attrib, values, label):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            ssv = []
            for s in samples:
                if (s[attrib] == v):
                    ssv.append(s[label])
            calc_entropy = self.measure(ssv)
            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy

class GiniIndex(InformationGain):
    def gain(self, samples, feature, feature_values, label):
        # Total Entropy
        ssv = []
        for s in samples:
            ssv.append(s[label])
        le = self.label_measure(ssv)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label)

        return le - ee

    def measure(self, samples):
        count = len(samples)
        label_counts = {}

        for s in samples:
            if s not in label_counts:
                label_counts[s] = 0
            label_counts[s] += 1
        
        res = 0
        
        for l in label_counts.values():
            prob = l / count
            res += prob * prob
        
        return 1 - res
    
    def label_measure(self, label_samples):
        count = len(label_samples)
        label_counts = {}

        for s in label_samples:
            if s not in label_counts:
                label_counts[s] = 0
            label_counts[s] += 1
        
        res = 0
        
        for l in label_counts.values():
            prob = l / count
            res += prob * prob
        
        return 1 - res
    
    def expected_measure(self, samples, attrib, values, label):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            ssv = []
            for s in samples:
                if (s[attrib] == v):
                    ssv.append(s[label])
            calc_entropy = self.measure(ssv)
            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy
