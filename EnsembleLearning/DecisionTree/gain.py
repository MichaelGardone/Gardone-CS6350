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
        ssv = samples[label].tolist()
        weights = samples["weight"].tolist()

        le = self.label_measure(ssv, weights)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label)

        return le - ee

    def measure(self, samples, weights):
        '''
        The equivalent of H({p1, p2,...,pk}) = -SUM(k, i=1)pi * log2(pi) .\n
        [Arg] sv -> sv is the sub-sample of examples. Should only be a list of length 2 Something else, and the label.
        '''
        sample_count = len(samples)

        # if we have 0 or 1 sample, then there is no entropy here
        if sample_count < 2:
            return 0

        # Feature Label Count
        fl_count = {-1:0, 1:0}
        
        for i in range(sample_count):
            fl_count[samples[i]] += weights[i]
        
        entropy = 0
        total = sum(weights)
        for key in fl_count:
            prob = fl_count[key] / total
            entropy += -prob * (math.log2(prob) if prob > 0.0 else 0)

        return entropy
    
    def label_measure(self, label_samples, weights):
        sample_count = len(label_samples)

        # if we have 0 or 1 sample, then there is no entropy here
        if sample_count < 2:
            return 0

        # Feature Label Count
        fl_count = {-1:0, 1:0}

        for i in range(sample_count):
            fl_count[label_samples[i]] += weights[i]
        
        entropy = 0
        total = sum(weights) # should be 1 as weights all sum to 1
        for key in fl_count:
            prob = fl_count[key] / total
            entropy += -prob * (math.log2(prob) if prob > 0.0 else 0)

        return entropy
    
    def expected_measure(self, samples, attrib, values, label):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            # where attrib in col == [v]alue, get the label
            ssv = samples.loc[samples[attrib] == v, label].tolist()
            weights = samples.loc[samples[attrib] == v, "weight"].tolist()
            
            calc_entropy = self.measure(ssv, weights)

            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy

class MajorityError(InformationGain):
    def gain(self, samples, feature, feature_values, label):
        # Total Entropy
        ssv = samples[label].tolist()
        weights = samples["weight"].tolist()

        le = self.label_measure(ssv, weights)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label, weights)

        return le - ee

    def measure(self, samples, weights):
        sample_count = len(samples)

        if sample_count == 0:
            return 0

        # Feature Label Count
        label_counts = {-1:0, 1:0}

        for i in range(sample_count):
            label_counts[samples[i]] += weights[i]

        res = label_counts[min(label_counts, key=label_counts.get)] / sample_count
   
        return res if res < 1.0 else 0
    
    def label_measure(self, label_samples, weights):
        sample_count = len(label_samples)

        # Feature Label Count
        label_counts = {-1:0, 1:0}

        for i in range(sample_count):
            label_counts[label_samples[i]] += weights[i]

        res = label_counts[min(label_counts, key=label_counts.get)] / sample_count
   
        return res if res < 1.0 else 0
    
    def expected_measure(self, samples, attrib, values, label, weights):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            ssv = samples.loc[samples[attrib] == v, label].tolist()
            calc_entropy = self.measure(ssv, weights)
            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy

class GiniIndex(InformationGain):
    def gain(self, samples, feature, feature_values, label):
        # Total Entropy
        ssv = samples[label].tolist()
        weights = samples["weight"].tolist()

        le = self.label_measure(ssv, weights)

        # Expected Entropy
        ee = self.expected_measure(samples, feature, feature_values, label, weights)

        return le - ee

    def measure(self, samples, weights):
        count = len(samples)
        tot_weight = sum(weights)
        label_counts = {-1:0, 1:0}

        for i in range(count):
            label_counts[samples[i]] += weights[i]
        
        res = 0
        
        for l in label_counts.values():
            prob = l / tot_weight
            res += prob * prob
        
        return 1 - res
    
    def label_measure(self, label_samples, weights):
        count = len(label_samples)
        tot_weight = sum(weights)
        label_counts = {-1:0, 1:0}

        for i in range(count):
            label_counts[label_samples[i]] += weights[i]
        
        res = 0
        
        for l in label_counts.values():
            prob = l / tot_weight
            res += prob * prob
        
        return 1 - res
    
    def expected_measure(self, samples, attrib, values, label, weights):
        count = len(samples)

        expected_entropy = 0
        for v in values:
            ssv = samples.loc[samples[attrib] == v, label].tolist()
            calc_entropy = self.measure(ssv, weights)
            expected_entropy += len(ssv) / count * calc_entropy

        return expected_entropy
