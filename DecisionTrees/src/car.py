from cgitb import reset
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

def print_results(depth, label, right, wrong, percent):
    output = "At depth " + str(depth) + " for " + label + "\n"
    output += "\t Total Right:" + str(right) + "\n"
    output += "\t Total Wrong:" + str(wrong) + "\n"
    output += "\t % Wrong:" + str(percent)
    print(output)

def main():
    train = read_train_csv()
    test = read_test_csv()

    test_total = len(test)
    train_total = len(train)

    entropy = gain.EntropyGain()
    majority_error = gain.MajorityError()
    gini_index = gain.GiniIndex()
    
    # Uncomment to print out a 6-depth tree!
    # car_entropy = id3.ID3("label", entropy, 6)
    
    # eroot = car_entropy.generate_tree(train, FEATURES)
    # car_entropy.print_tree(eroot)
    
    results = {}

    for i in range(1, 7):
        res = { "gi":{ "training":{}, "testing":{} }, "e":{ "training":{}, "testing":{} }, "me":{ "training":{}, "testing":{} } }

        car_entropy = id3.ID3("label", entropy, i)
        e_root = car_entropy.generate_tree(train, FEATURES)

        car_me = id3.ID3("label", majority_error, i)
        me_root = car_me.generate_tree(train, FEATURES)

        car_gi = id3.ID3("label", gini_index, i)
        gi_root = car_gi.generate_tree(train, FEATURES)

        wrong = 0
        for j in range(len(test)):
            if car_entropy.predict(e_root, test[j]) != test[j]["label"]:
                wrong += 1

        res["e"]["testing"]["right"] = (test_total - wrong)
        res["e"]["testing"]["wrong"] = wrong
        res["e"]["testing"]["perc"] = (wrong / test_total)

        wrong = 0
        for j in range(len(train)):
            if car_entropy.predict(e_root, train[j]) != train[j]["label"]:
                wrong += 1
        
        res["e"]["training"]["right"] = (train_total - wrong)
        res["e"]["training"]["wrong"] = wrong
        res["e"]["training"]["perc"] = (wrong / train_total)
        
        wrong = 0
        for j in range(len(test)):
            if car_me.predict(me_root, test[j]) != test[j]["label"]:
                wrong += 1
        
        res["me"]["testing"]["right"] = (test_total - wrong)
        res["me"]["testing"]["wrong"] = wrong
        res["me"]["testing"]["perc"] = (wrong / test_total)

        wrong = 0
        for j in range(len(train)):
            if car_me.predict(me_root, train[j]) != train[j]["label"]:
                wrong += 1
        
        res["me"]["training"]["right"] = (train_total - wrong)
        res["me"]["training"]["wrong"] = wrong
        res["me"]["training"]["perc"] = (wrong / train_total)

        wrong = 0
        for j in range(len(test)):
            if car_gi.predict(gi_root, test[j]) != test[j]["label"]:
                wrong += 1
        
        res["gi"]["testing"]["right"] = (test_total - wrong)
        res["gi"]["testing"]["wrong"] = wrong
        res["gi"]["testing"]["perc"] = (wrong / test_total)

        wrong = 0
        for j in range(len(train)):
            if car_gi.predict(gi_root, train[j]) != train[j]["label"]:
                wrong += 1
        
        res["gi"]["training"]["right"] = (train_total - wrong)
        res["gi"]["training"]["wrong"] = wrong
        res["gi"]["training"]["perc"] = (wrong / train_total)
        
        results[i] = res
    
    for m in ["gi", "e", "me"]:
        if m == "gi":
            print("==== GINI INDEX ====")
        elif m == "e":
            print("==== ENTROPY ====")
        elif m == "me":
            print("==== MAJORITY ERROR ====")
        for i in range(1,7):
            print_results(i, "training", results[i][m]["training"]["right"], results[i][m]["training"]["wrong"], results[i][m]["training"]["perc"])
            print_results(i, "testing", results[i][m]["testing"]["right"], results[i][m]["testing"]["wrong"], results[i][m]["testing"]["perc"])
        if m == "gi":
            print("==== GINI INDEX ====")
        elif m == "e":
            print("==== ENTROPY ====")
        elif m == "me":
            print("==== MAJORITY ERROR ====")


if __name__ == "__main__":
    main()
