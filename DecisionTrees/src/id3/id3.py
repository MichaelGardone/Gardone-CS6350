from copy import deepcopy
from operator import itemgetter

class Node(object):
    def __init__(self) -> None:
        # What does this node represent?
        self.label = None
        # Branches are other Node objects
        self.branches = []
        # Depth
        self.depth = 0
        # Leaf status
        self.is_leaf = False
    
    def add_branch(self, connection, condition) -> None:
        self.branches.append([connection, condition])
    
    def set_label(self, label):
        self.label = label
    
    def get_is_leaf(self) -> bool:
        return self.is_leaf
    
    def set_leaf(self):
        self.is_leaf = True

class ID3:
    def __init__(self, label, gain, max_depth=6) -> None:
        self.label = label

        # Entropy, Majority Error, or Gini Index
        self.gain = gain

        # The maximum the tree 
        self.max_depth = max_depth

        # Root node
        self.root = None
    
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

        if depth == 1:
            self.root = node

        # if all examples have same label, return label
        if self.are_labels_same(examples):
            node.set_leaf()
            node.set_label(examples[0][self.label])
            return node

        # if attribs is empty or we hit max depth, return a leaf node with the most common label
        if depth >= self.max_depth or len(attribs) == 0:
            node.set_leaf()
            node.set_label(self.get_most_common_label(examples))
            return node

        # A = attribute in Attributes that best splits S
        gains = []
        for a in attribs.keys():
            gains.append([self.gain.gain(examples, a, attribs[a], self.label), a])
        A = max(gains, key=itemgetter(0))[1]

        node.set_label(A)

        for val in attribs[A]:
            sv = self.get_subset(examples, A, val)
            if len(sv) == 0: # 0, because get_subset will return a list of length 0 if nothing fits
                child = Node()
                child.set_leaf()
                child.set_label(self.get_most_common_label(examples))
                node.add_branch(child, val)
                child.depth = depth + 1
                return node
            else:
                av = deepcopy(attribs)
                av.pop(A)
                node.add_branch(self.generate_tree(sv, av, depth + 1), val)
        
        return node

    def predict(self, sample) -> str:
        currNode = self.root

        while currNode != None and currNode.get_is_leaf() == False:
            branch_on = sample[currNode.label]
            currNode = self._branch_on(currNode, branch_on)

            if currNode == None:
                break

            if currNode.get_is_leaf():
                return currNode.label

        if currNode == self.root:
            return self.root.label

        return None

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
    
    def get_root(self) -> Node:
        return self.root

    def print_tree(self, cn):
        print(cn.label)

        for i in range(len(cn.branches)):
            self._print_tree(cn.branches[i])

    def _print_tree(self, cn):
        node = cn[0]
        cond = cn[1]
        out = ("\t" * (node.depth - 1)) + '[on ' + cond +']'+ node.label
        print(out)

        for i in range(len(node.branches)):
            self._print_tree(node.branches[i])
    
    def _branch_on(self, node, branch_on):
        for e in node.branches:
            if e[1] == branch_on:
                return e[0]
        return None
