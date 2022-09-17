from id3 import id3, gain
import statistics

LABELS = ["yes", "no"]

COLUMNS = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]

FEATURES = {
    "age":          ["T", "F"],
    "job":          ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    "marital":      ["married","divorced","single"],
    "education":    ["unknown","secondary","primary","tertiary"],
    "default":      ["yes","no"],
    "balance":      ["T", "F"],
    "housing":      ["yes","no"],
    "loan":         ["yes","no"],
    "contact":      ["unknown","telephone","cellular"],
    "day":          ["T", "F"],
    "month":        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration":     ["T", "F"],
    "campaign":     ["T", "F"],
    "pdays":        ["T", "F"],
    "previous":     ["T", "F"],
    "poutcome":     ["unknown","other","failure","success"]
}

def read_train_csv(replace=False):
    global COLUMNS, FEATURES

    # A sample looks like this:
    # [ ..., { "buying":VAL, "maint":VAL, ..., "label":VA L} ,... ]
    samples = []
    cols_with_unk = []

    with open("../bank/train.csv", 'r') as f:
        for line in f:
            raw_sample = line.strip().split(',')
            sample = {}
            for i in range(len(raw_sample)):
                if COLUMNS[i] in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
                    sample[COLUMNS[i]] = float(raw_sample[i])
                else:
                    sample[COLUMNS[i]] = raw_sample[i]
                    if raw_sample[i] == "unknown" and COLUMNS[i] not in cols_with_unk:
                        cols_with_unk.append(COLUMNS[i])
            samples.append(sample)
    
    # Convert int/float vals to binary values
    labels = ["T", "F"]
    for f in FEATURES.keys():
        if FEATURES[f] == ["T", "F"]:
            FEATURES[f] = labels
            numbers = []
            
            for s in samples:
                numbers.append(s[f])
            
            median = statistics.median(numbers)

            for s in samples:
                s[f] = "F" if s[f] < median else "T"

    # if replace is true, we need to replace "unknown" with the most common label in the data so far    
    if replace:
        for c in cols_with_unk:
            data = {}
            for s in samples:
                if s[c] != "unknown":
                    if s[c] not in data:
                        data[s[c]] = 0
                    data[s[c]] += 1
            
            replace = max(data, key=data.get)
            
            for s in samples:
                if s[c] == "unknown":
                    s[c] = replace

    return samples

def read_test_csv(replace=False):
    global COLUMNS

    # A sample looks like this:
    # [ ..., { "buying":VAL, "maint":VAL, ..., "label":VA L} ,... ]
    samples = []
    cols_with_unk = []

    with open("../bank/test.csv", 'r') as f:
        for line in f:
            raw_sample = line.strip().split(',')
            sample = {}
            for i in range(len(raw_sample)):
                if COLUMNS[i] in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
                    sample[COLUMNS[i]] = float(raw_sample[i])
                else:
                    sample[COLUMNS[i]] = raw_sample[i]
                    if raw_sample[i] == "unknown" and COLUMNS[i] not in cols_with_unk:
                        cols_with_unk.append(COLUMNS[i])
            samples.append(sample)
    
    # Convert int/float vals to binary values
    labels = ["T", "F"]
    for f in FEATURES.keys():
        if FEATURES[f] == labels:
            numbers = []
            
            for s in samples:
                numbers.append(s[f])
            
            median = statistics.median(numbers)

            for s in samples:
                s[f] = "F" if s[f] < median else "T"

    # if replace is true, we need to replace "unknown" with the most common label in the data so far    
    if replace:
        for c in cols_with_unk:
            data = {}
            for s in samples:
                if s[c] != "unknown":
                    if s[c] not in data:
                        data[s[c]] = 0
                    data[s[c]] += 1
            
            replace = max(data, key=data.get)
            
            for s in samples:
                if s[c] == "unknown":
                    s[c] = replace
    
    return samples

def main():
    train = read_train_csv()
    test = read_test_csv()
    train_wo_unk = read_train_csv(True)
    test_wo_unk = read_test_csv(True)

    entropy = gain.EntropyGain()
    majority_error = gain.MajorityError()
    gini_index = gain.GiniIndex()

    test_total = len(test)
    train_total = len(train)

    print("============== UNKNOWN AS VALUE ==============")
    print("===== ENTROPY =====")

    for i in range(1, 17):
        bank = id3.ID3("y", entropy, i)
        root = bank.generate_tree(train, FEATURES)

        # print("\n")

        wrong = 0
        for j in range(len(test)):
            if bank.predict(root, test[j]) != test[j]["y"]:
                wrong += 1
        
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train)):
            if bank.predict(root, train[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== ENTROPY =====")
    
    print("===== MAJORITY ERROR =====")
    
    for i in range(1, 17):
        bank = id3.ID3("y", majority_error, i)
        root = bank.generate_tree(train, FEATURES)

        wrong = 0

        for j in range(len(test)):
            if bank.predict(root, test[j]) != test[j]["y"]:
                wrong += 1
        
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train)):
            if bank.predict(root, train[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== MAJORITY ERROR =====")

    print("===== GINI INDEX =====")
    
    for i in range(1, 17):
        bank = id3.ID3("y", gini_index, i)
        root = bank.generate_tree(train, FEATURES)

        wrong = 0

        for j in range(len(test)):
            if bank.predict(root, test[j]) != test[j]["y"]:
                wrong += 1
        
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train)):
            if bank.predict(root, train[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== GINI INDEX =====")
    print("============== UNKNOWN AS VALUE ==============")
    
    print()

    print("============== UNKNOWN REPLACED ==============")
    print("===== ENTROPY =====")

    for i in range(1, 17):
        bank = id3.ID3("y", entropy, i)
        root = bank.generate_tree(train_wo_unk, FEATURES)

        wrong = 0
        for j in range(len(test_wo_unk)):
            if bank.predict(root, test_wo_unk[j]) != test[j]["y"]:
                wrong += 1
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train_wo_unk)):
            if bank.predict(root, train_wo_unk[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== ENTROPY =====")
    
    print("===== MAJORITY ERROR =====")
    
    for i in range(1, 17):
        bank = id3.ID3("y", majority_error, i)
        root = bank.generate_tree(train_wo_unk, FEATURES)

        wrong = 0
        for j in range(len(test_wo_unk)):
            if bank.predict(root, test_wo_unk[j]) != test[j]["y"]:
                wrong += 1
        
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train_wo_unk)):
            if bank.predict(root, train_wo_unk[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== MAJORITY ERROR =====")

    print("===== GINI INDEX =====")
    
    for i in range(1, 17):
        bank = id3.ID3("y", gini_index, i)
        root = bank.generate_tree(train_wo_unk, FEATURES)

        wrong = 0
        for j in range(len(test_wo_unk)):
            if bank.predict(root, test_wo_unk[j]) != test[j]["y"]:
                wrong += 1
        
        print("Test @ Depth", i, ":", wrong, "(", (wrong / test_total),")")
        print()

        wrong = 0
        for j in range(len(train_wo_unk)):
            if bank.predict(root, train_wo_unk[j]) != train[j]["y"]:
                wrong += 1
        
        print("Train @ Depth", i, ":", wrong, "(", (wrong / train_total),")")
        print()
    
    print("===== GINI INDEX =====")
    print("============== UNKNOWN REPLACED ==============")


if __name__ == "__main__":
    main()
