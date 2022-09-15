
class Node(object):
    def __init__(self) -> None:
        # What does this node represent?
        self.label = None
        # Branches are other Node objects
        self.branches = []
        # Depth
        self.depth = 0
    
    def add_branch(self, edge) -> None:
        self.branches.append(edge)

class ID3:
    def __init__(self, label, gain, max_depth=6) -> None:
        self.label = label

        # Entropy, Majority Error, or Gini Index
        self.gain = gain

        self.max_depth = max_depth
    
    def get_subset(self, examples, attribute, value) -> list:
        subset = []

        for e in examples:
            if e[attribute] == value:
                pruned_e = {}
                for key,val in e.items():
                    if attribute == key: continue
                    pruned_e[key] = val
                subset.append(pruned_e)

        return subset
    
    def generate_tree(self, examples, attribs) -> Node:
        
        node = Node()

        # A = attribute in Attributes that best splits S
        gains = []
        for a in attribs.keys():
            gains.append([self.gain(examples, a, attribs[a], self.label), a])
        A = max(gains, key=lambda g:gains[0])[1]

    def predict(self, sample) -> str:
        pass

def get_subset(examples, attribute, value) -> list:
    subset = []

    for e in examples:
        if e[attribute] == value:
            pruned_e = {}
            for key,val in e.items():
                if attribute == key: continue
                pruned_e[key] = val
            subset.append(pruned_e)

    return subset

def ID3(examples, attribs, labels, gain) -> Node:
    '''Generate a decision tree based on the ID3 algorithm.\n
    [Arg] S:
    '''
    # Did we reach a leaf?
    # All positive or negative - return label

    node = None
    
    A = get_subset(examples, attribs)

    return node
