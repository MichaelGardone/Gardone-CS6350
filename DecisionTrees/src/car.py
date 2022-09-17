from id3 import id3, gain

LABELS = ["unacc", "acc", "good", "vgood"]

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]

FEATURES = { "buying":   ["vhigh", "high", "med", "low"],
             "maint":    ["vhigh", "high", "med", "low"],
             "doors":    ["2", "3", "4", "5more"],
             "persons":  ["2", "4", "more"],
             "lug_boot": ["small", "med", "big"],
             "safety":   ["low", "med", "high"]
        }

def read_train_csv():
    global COLUMNS

    # A sample looks like this:
    # [ ..., { "buying":VAL, "maint":VAL, ..., "label":VA L} ,... ]
    samples = []

    with open("../car/train.csv", 'r') as f:
        for line in f:
            # Produces: [low, vhigh, 4, 4, big, med, acc]
            # samples.append(line.strip().split(','))
            raw_sample = line.strip().split(',')
            sample = {}
            for i in range(len(raw_sample)):
                sample[COLUMNS[i]] = raw_sample[i]
            samples.append(sample)
    
    return samples

def read_test_csv():
    global COLUMNS

    # A sample looks like this:
    # [ ..., { "buying":VAL, "maint":VAL, ..., "label":VA L} ,... ]
    samples = []

    with open("../car/test.csv", 'r') as f:
        for line in f:
            # Produces: [low, vhigh, 4, 4, big, med, acc]
            # samples.append(line.strip().split(','))
            raw_sample = line.strip().split(',')
            sample = {}
            for i in range(len(raw_sample)):
                sample[COLUMNS[i]] = raw_sample[i]
            samples.append(sample)
    
    return samples

def main():
    train = read_train_csv()
    test = read_test_csv()

    test_total = len(test)
    train_total = len(train)

    entropy = gain.EntropyGain()
    majority_error = gain.MajorityError()
    gini_index = gain.GiniIndex()
    
    print("===== ENTROPY =====")

    for i in range(1, 7):
        car = id3.ID3("label", entropy, i)
        car.generate_tree(train, FEATURES)

        print("=== LIMITED TO", i, "===")
        # print("\n")

        wrong = 0
        
        print("= Test Dataset Results =")
        for j in range(len(test)):
            if car.predict(test[j]) != test[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / test_total))
        print()

        wrong = 0
        print("= Training Dataset Results =")
        for j in range(len(train)):
            if car.predict(train[j]) != train[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / train_total))
        print()
    
    print("===== ENTROPY =====")
    
    print("===== MAJORITY ERROR =====")
    
    for i in range(1, 7):
        car = id3.ID3("label", majority_error, i)
        car.generate_tree(train, FEATURES)

        print("=== LIMITED TO", i, "===")
        # print("\n")

        wrong = 0

        print("= Test Dataset Results =")
        for j in range(len(test)):
            if car.predict(test[j]) != test[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / test_total))
        print()

        wrong = 0
        print("= Training Dataset Results=")
        for j in range(len(train)):
            if car.predict(train[j]) != train[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / train_total))
        print()
    
    print("===== MAJORITY ERROR =====")

    print("===== GINI INDEX =====")
    
    for i in range(1, 7):
        car = id3.ID3("label", gini_index, i)
        car.generate_tree(train, FEATURES)

        print("=== LIMITED TO", i, "===")
        # print("\n")

        wrong = 0

        print("= Test Dataset Results =")
        for j in range(len(test)):
            if car.predict(test[j]) != test[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / test_total))
        print()

        wrong = 0
        print("= Training Dataset Results =")
        for j in range(len(train)):
            if car.predict(train[j]) != train[j]["label"]:
                wrong += 1
        
        print("Total Wrong:", wrong)
        print("% Wrong:", (wrong / train_total))
        print()
    
    print("===== GINI INDEX =====")

if __name__ == "__main__":
    main()
