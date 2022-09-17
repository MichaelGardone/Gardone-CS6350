from copy import deepcopy
from operator import itemgetter

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

class ID3:
    def __init__(self, label, gain, max_depth=6) -> None:
        self.label = label

        # Entropy, Majority Error, or Gini Index
        self.gain = gain

        # The maximum the tree 
        self.max_depth = max_depth

        # Root node
        self.root = None

        # self.seen = False
    
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
    
    def generate_tree(self, examples, attribs, depth=1) -> Node:
        node = Node()
        node.depth = depth

        # if all examples have same label, return label
        if self.are_labels_same(examples):
            node.set_label(examples[0][self.label])
            return node

        # if attribs is empty or we hit max depth, return a leaf node with the most common label
        if depth > self.max_depth or len(attribs) == 0:
            node.set_label(self.get_most_common_label(examples))
            return node

        # A = attribute in Attributes that best splits S
        gains = {}
        for a in attribs.keys():
            gains[a] = self.gain.gain(examples, a, attribs[a], self.label)
        A = max(gains, key=gains.get)

        node.set_label(A)

        # if A == "doors" and self.seen == False:
        #     print(attribs[A])
        #     for val in attribs[A]:
        #         sv = self.get_subset(examples, A, val)
        #         print(val, len(sv), sv)
        #     self.seen = True

        for val in attribs[A]:
            sv = self.get_subset(examples, A, val)
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

    def predict(self, node, sample) -> str:
        if node.is_leaf():
            return node.label
        else:
            # print(node.label, "branching on", sample[node.label])
            return self.predict(node.branches[sample[node.label]], sample)


    def are_labels_same(self, examples):
        labels = []

        for e in examples:
            if e[self.label] not in labels:
                labels.append(e[self.label])
        
        return len(labels) == 1
    
    def get_most_common_label(self, examples):
        count = {}

        for e in examples:
            if e[self.label] not in count.keys():
                count[e[self.label]] = 0
            count[e[self.label]] += 1
        
        return max(count, key=count.get)

    def print_tree(self, cn):
        print(cn.label)

        for cond, node in cn.branches.items():
            self._print_tree(cond, node)

    def _print_tree(self, cond, node):
        out = ("\t" * node.depth) + '[on ' + cond +']'+ str(node.label) + " (children " + str(len(node.branches)) + ")" 
        print(out)

        for cond, node in node.branches.items():
            self._print_tree(cond, node)
    
    def _branch_on(self, node, branch_on):
        for e in node.branches:
            if e[1] == branch_on:
                return e[0]
        return None
