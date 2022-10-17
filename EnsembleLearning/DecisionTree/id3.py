import random

class Node(object):
    def __init__(self) -> None:
        # What does this node represent?
        self.label = None
        # Branches are other Node objects
        self.branches = {}
        # Depth
        self.depth = 0
    
    def add_branch(self, to, condition) -> None:
        self.branches[condition] = to
    
    def set_label(self, label):
        self.label = label
    
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    def print_tree(self):
        print(self.label)

        for cond, node in self.branches.items():
            self._print_tree(cond, node)

    def _print_tree(self, cond, node):
        out = ("\t" * node.depth) + '[on ' + str(cond) +']'+ str(node.label) + " (children " + str(len(node.branches)) + ")" 
        print(out)

        for cond, node in node.branches.items():
            self._print_tree(cond, node)
    
    def predict(self, sample) -> str:
        if self.is_leaf():
            return self.label
        else:
            return self.branches[sample[self.label]].predict(sample)

class EID3:
    def __init__(self, label, gain) -> None:
        self.label = label

        # Entropy, Majority Error, or Gini Index
        self.gain = gain

        # Root node
        self.root = None
    
    def get_subset(self, examples, attribute, attributes, value) -> list:
        all_but_self = [a for a in attributes if a != attribute]
        all_but_self.append(self.label)
        all_but_self.append("weight")

        # subset = examples.loc[examples[attribute] == value, all_but_self]
        # print(subset)
        # print(subset.columns.tolist())

        return examples.loc[examples[attribute] == value, all_but_self]
    
    def generate_tree_with_subset(self, examples, attribs, attrib_count = 1, depth = 0) -> Node:
        node = Node()
        node.depth = depth

        # if all examples have same label, return label
        if self.are_labels_same(examples):
            node.set_label(examples[self.label].tolist()[0])
            return node

        # if attribs is empty, return a leaf node with the most common label
        if len(attribs) == 0:
            node.set_label(self.get_most_common_label(examples))
            return node
        
        # Get G from A

        # if # of attribs requested is greater than what is available, just take the available amount
        act_attrib_count = attrib_count
        if attrib_count > len(attribs):
            act_attrib_count = len(attribs)

        sub_attribs = {}
        a = list(attribs)
        while len(sub_attribs) < act_attrib_count:
            choice = random.choice(a)
            if choice not in sub_attribs:
                sub_attribs[choice] = attribs[choice]

        # A = attribute in Attributes that best splits S
        gains = {}
        for a in sub_attribs.keys():
            gains[a] = self.gain.gain(examples, a, sub_attribs[a], self.label)
        A = max(gains, key=gains.get)

        node.set_label(A)

        for val in attribs[A]:
            sv = self.get_subset(examples, A, attribs, val)
            if len(sv) == 0:
                child = Node()
                child.set_label(self.get_most_common_label(examples))
                node.add_branch(child, val)
                child.depth = depth + 1
            else:
                av = {}
                for k,v in attribs.items():
                    if k != A:
                        av[k] = v
                
                if len(av) > 0:
                    node.add_branch(self.generate_tree(sv, av, depth + 1), val)
        
        return node

    def generate_tree(self, examples, attribs, depth = 0) -> Node:
        node = Node()
        node.depth = depth

        # if all examples have same label, return label
        if self.are_labels_same(examples):
            node.set_label(examples[self.label].tolist()[0])
            return node

        # if attribs is empty, return a leaf node with the most common label
        if len(attribs) == 0:
            node.set_label(self.get_most_common_label(examples))
            return node

        # A = attribute in Attributes that best splits S
        gains = {}
        for a in attribs.keys():
            gains[a] = self.gain.gain(examples, a, attribs[a], self.label)
        A = max(gains, key=gains.get)

        node.set_label(A)

        for val in attribs[A]:
            sv = self.get_subset(examples, A, attribs, val)
            if len(sv) == 0:
                child = Node()
                child.set_label(self.get_most_common_label(examples))
                node.add_branch(child, val)
                child.depth = depth + 1
            else:
                av = {}
                for k,v in attribs.items():
                    if k != A:
                        av[k] = v
                
                node.add_branch(self.generate_tree(sv, av, depth + 1), val)
        
        return node
    
    def generate_stump(self, examples, attribs) -> Node:
        node = Node()
        node.depth = 0

        # A = attribute in Attributes that best splits S
        gains = {}
        for a in attribs.keys():
            gains[a] = self.gain.gain(examples, a, attribs[a], self.label)
        A = max(gains, key=gains.get)

        # set the node to be the best label A
        node.set_label(A)

        # Go through every possible value of the attribute and see what it says it should be based
        #   on the most common label
        for val in attribs[A]:
            sv = self.get_subset(examples, A, attribs, val)
            child = Node()
            child.set_label(self.get_most_common_label(sv))
            child.depth = 1
            node.add_branch(child, val)
        
        return node

    def are_labels_same(self, examples):
        labels = []
        labels_from_examples = examples[self.label].tolist()

        for e in labels_from_examples:
            if e not in labels:
                labels.append(e)
        
        return len(labels) == 1
    
    def get_most_common_label(self, examples):
        count = {-1:0, 1:0}

        ssv = examples[self.label].tolist()
        weights = examples["weight"].tolist()

        for i in range(len(ssv)):
            count[ssv[i]] += weights[i]
        
        return max(count, key=count.get)
