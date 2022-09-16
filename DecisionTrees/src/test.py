from random import sample
import play
import math
from copy import deepcopy
from id3 import gain

# print("==================== INIT TEST ====================")
samples = play.read_csv()

# Total count; train is 1000 samples
# print("Count:", len(samples))

# print(samples)

# counts = {}

# for s in samples:
#     for i in range(len(s)):
#         if s[i] not in counts:
#             counts[s[i]] = 1
#         else:
#             counts[s[i]] += 1

# for key,val in counts.items():
#     print(key, ":", val)

#============= Subset Test ===============
# print("\n\n\n")
# print("==================== SUBSET TEST ====================")

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

# subset = get_subset(samples, 'outlook', 'S')

# print("Subset Count:", len(subset), " == 5 ; if not, problem!")
# print("Subset Elem Count:", len(subset[0]), " == 4 ; if not, problem!")
# print(subset)

# subset = get_subset(subset, 'temperature', 'H')

# print("Subset Count:", len(subset), " == 2 ; if not, problem!")
# print("Subset Elem Count:", len(subset[0]), " == 3 ; if not, problem!")
# print(subset)

#============= Entropy Test ===============
# print("\n\n\n")
# print("==================== ENTROPY TEST ====================")

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

# sv = []

# for s in samples:
#     if (s["outlook"] == "S"):
#         sv.append([s["outlook"], s["play"]])

# entropy_sunny = entropy(sv)

# sv = []

# for s in samples:
#     if (s["outlook"] == "O"):
#         sv.append([s["outlook"], s["play"]])

# entropy_overcast = entropy(sv)

# sv = []

# for s in samples:
#     if (s["outlook"] == "R"):
#         sv.append([s["outlook"], s["play"]])

# entropy_rain = entropy(sv)

# print("Entropy for outlook=S:", entropy_sunny)
# print("Entropy for outlook=O:", entropy_overcast)
# print("Entropy for outlook=R:", entropy_rain)

# print("Expected Entropy:", ((5/14) * entropy_sunny + (5/14) * entropy_rain))

def expected_entropy(sv, attrib, values, label):
    count = len(sv)

    if count == 0:
        return 0

    expected_entropy = 0
    for v in values:
        ssv = []
        for s in sv:
            if (s[attrib] == v):
                ssv.append([s[attrib], s[label]])
        calc_entropy = entropy(ssv)
        expected_entropy += len(ssv) / count * calc_entropy

    return expected_entropy

# ee = expected_entropy(samples, "outlook", play.FEATURES["outlook"], play.COLUMNS[-1])
# print("Expected Entropy (from expected_entropy function):", ee)

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

# sv = []

# for s in samples:
#     sv.append(s["play"])

# le = label_entropy(sv)
# print("Label Entropy:", le)

# print("Information Gain:", (le - ee))

def information_gain(sv, feature, feature_values, label):
    # Total Entropy
    ssv = []
    for s in sv:
        ssv.append(s[label])
    le = label_entropy(ssv)

    # Expected Entropy
    ee = expected_entropy(sv, feature, feature_values, label)

    return le - ee

# ig = information_gain(samples, "outlook", play.FEATURES["outlook"], play.COLUMNS[-1])
# print("Information Gain (from information_gain function):", ig)

# for key,val in fl_count.items():
#     print(key, ":", val)


# 
# print("\n\n\n")
# print("==================== SPLIT TEST ====================")
# gains = []
# for a in play.FEATURES.keys():
#     gains.append([information_gain(samples, a, play.FEATURES[a], "play"), a])
# print(gains)
# A = max(gains, key=lambda g:gains[0])[1]
# print(A)

# print("\n\n\n")
# print("==================== MCL TEST ====================")

def get_most_common_label(examples, label_col):
    count = {}

    for e in examples:
        if e[label_col] not in count.keys():
            count[e[label_col]] = 0
        count[e[label_col]] += 1
    
    return max(count, key=count.get)

# mcl = get_most_common_label(samples, "play")
# print("Most common label:", mcl, "should be +")

# print("\n\n\n\n\n\n\n\n\n\n\n")
print("==================== TEST ====================")

# def are_labels_same(examples):
#     labels = []

#     for e in examples:
#         if e["play"] not in labels:
#             labels.append(e["play"])
            
#     return len(labels) == 1

# gains = []
# for a in play.FEATURES.keys():
#     gains.append([information_gain(samples, a, play.FEATURES[a], "play"), a])
# print(gains)
# A = max(gains, key=lambda g:gains[0])[1]
# print("split on", A)

# atrb = deepcopy(play.FEATURES)
# atrb.pop(A)

# sssv = get_subset(samples, A, 'S')
# print(len(sssv))
# print(sssv)

# sssv = get_subset(samples, A, 'O')
# print(len(sssv))
# print(sssv)

# sssv = get_subset(samples, A, 'R')
# print(len(sssv))
# print(sssv)

# def majority_error(sv):
#     count = len(sv)
#     label_counts = {}

#     for s in sv:
#         if s not in label_counts:
#             label_counts[s] = 0
#         label_counts[s] += 1
        
#     res = label_counts[min(label_counts, key=label_counts.get)] / count
   
#     return res if res != 1 else 0

# sv = []
# for s in samples:
#     sv.append(s["play"])

# print(majority_error(sv), "==", (5/14))

# me = gain.MajorityError()

# print(me.measure(sv), "==", majority_error(sv), "==", (5/14))

# sv = []
# for s in samples:
#     if s["outlook"] == "S":
#         sv.append(s["play"])

# ee_s = majority_error(sv)
# print(ee_s)
# ee_s *= len(sv) / len(samples)
# print("/",ee_s)


# sv = []
# for s in samples:
#     if s["outlook"] == "O":
#         sv.append(s["play"])

# ee_o = majority_error(sv)
# # print(ee_o)
# ee_o *= len(sv) / len(samples)

# sv = []
# for s in samples:
#     if s["outlook"] == "R":
#         sv.append(s["play"])

# ee_r = majority_error(sv)
# print("/",ee_r)
# ee_r *= len(sv) / len(samples)
# print("/",ee_r)

# print(ee_s + ee_r + ee_o)
# print((5/14) - (ee_s + ee_r + ee_o))

# gains = []
# for a in play.FEATURES.keys():
#     gains.append([me.gain(samples, a, play.FEATURES[a], "play"), a])
# print(gains)
# A = max(gains, key=lambda g:gains[0])[1]
# print("split on", A)

def gini_index(sv):
    count = len(sv)
    label_counts = {}

    for s in sv:
        if s not in label_counts:
            label_counts[s] = 0
        label_counts[s] += 1
    
    res = 0
    
    for l in label_counts.values():
        prob = l / count
        res += prob * prob
    
    return 1 - res

sv = []
for s in samples:
    sv.append(s["play"])

print(gini_index(sv), "==", (1 - ((5/14) * (5/14)) - ((9/14) * (9/14))))

me = gain.GiniIndex()

print(me.measure(sv), "==", gini_index(sv), "==", (1 - ((5/14) * (5/14)) - ((9/14) * (9/14))))

sv = []
for s in samples:
    if s["outlook"] == "S":
        sv.append(s["play"])

ee_s = gini_index(sv)
print(ee_s)
ee_s *= len(sv) / len(samples)
print("/",ee_s)

sv = []
for s in samples:
    if s["outlook"] == "O":
        sv.append(s["play"])

ee_o = gini_index(sv)
# print(ee_o)
ee_o *= len(sv) / len(samples)

sv = []
for s in samples:
    if s["outlook"] == "R":
        sv.append(s["play"])

ee_r = gini_index(sv)
print("/",ee_r)
ee_r *= len(sv) / len(samples)
print("/",ee_r)

print(ee_s + ee_r + ee_o)
print((5/14) - (ee_s + ee_r + ee_o))

gains = []
for a in play.FEATURES.keys():
    gains.append([me.gain(samples, a, play.FEATURES[a], "play"), a])
print(gains)
A = max(gains, key=lambda g:gains[0])[1]
print("split on", A)
