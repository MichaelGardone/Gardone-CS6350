import play
from id3 import gain
from id3 import id3

print("==================== INIT TEST ====================")
samples = play.read_csv()

# Total count; train is 1000 samples
print("Count:", len(samples))

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
print("\n\n\n")
print("==================== SUBSET TEST ====================")
subset = id3.get_subset(samples, 'outlook', 'S')

print("Subset Count:", len(subset), " == 5 ; if not, problem!")
print("Subset Elem Count:", len(subset[0]), " == 4 ; if not, problem!")
print(subset)

subset = id3.get_subset(subset, 'temperature', 'H')

print("Subset Count:", len(subset), " == 2 ; if not, problem!")
print("Subset Elem Count:", len(subset[0]), " == 3 ; if not, problem!")
print(subset)

#============= Entropy Test ===============
print("\n\n\n")
print("==================== ENTROPY TEST ====================")

sv = []

for s in samples:
    if (s["outlook"] == "S"):
        sv.append([s["outlook"], s["play"]])
    # sv.append([s["buying"], s["label"]])

# print(sv)
# entropy, fl_count = gain.entropy(sv)
entropy_sunny = gain.entropy(sv)

sv = []

for s in samples:
    if (s["outlook"] == "O"):
        sv.append([s["outlook"], s["play"]])

entropy_overcast = gain.entropy(sv)

sv = []

for s in samples:
    if (s["outlook"] == "R"):
        sv.append([s["outlook"], s["play"]])

entropy_rain = gain.entropy(sv)

print("Entropy for outlook=S:", entropy_sunny)
print("Entropy for outlook=O:", entropy_overcast)
print("Entropy for outlook=R:", entropy_rain)

print("Expected Entropy:", ((5/14) * entropy_sunny + (5/14) * entropy_rain))

expected_entropy = gain.expected_entropy(samples, "outlook", play.FEATURES["outlook"], play.COLUMNS[-1])
print("Expected Entropy (from expected_entropy function):", expected_entropy)

sv = []

for s in samples:
    sv.append(s["play"])

label_entropy = gain.label_entropy(sv)
print("Label Entropy:", label_entropy)

print("Information Gain:", (label_entropy - expected_entropy))

information_gain = gain.information_gain(samples, "outlook", play.FEATURES["outlook"], play.COLUMNS[-1])
print("Information Gain (from information_gain function):", information_gain)

# for key,val in fl_count.items():
#     print(key, ":", val)


# 
print("\n\n\n")
print("==================== SPLIT TEST ====================")
gains = []
for a in play.FEATURES.keys():
    gains.append([gain.information_gain(samples, a, play.FEATURES[a], "play"), a])
print(gains)
A = max(gains, key=lambda g:gains[0])[1]
print(A)
