from id3 import id3, gain

LABELS = ["+", "-"]

COLUMNS = ["outlook", "temperature", "humidity", "wind", "play"]

FEATURES = { "outlook":     ['S', 'O', 'R'],
             "temperature": ['H', 'M', 'C'],
             "humidity":    ['H', 'N', 'L'],
             "wind":        ['S', 'W']
             # remove the Play column just because it shouldn't be calculated
        }

def read_csv():
    global COLUMNS

    # A sample looks like this:
    # [ ..., { "buying":VAL, "maint":VAL, ..., "label":VA L} ,... ]
    samples = []

    with open("../play/train.csv", 'r') as f:
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
    S = read_csv()

    entropy = gain.EntropyGain()
    majority_error = gain.MajorityError()
    gini_index = gain.GiniIndex()

    print("=== Entropy ===")

    play_entropy = id3.ID3("play", entropy)
    play_entropy.generate_tree(S, FEATURES)
    play_entropy.print_tree(play_entropy.get_root())

    print("\n\n")
    print("=== Majority Error ===")

    play_me = id3.ID3("play", majority_error)
    play_me.generate_tree(S, FEATURES)
    play_me.print_tree(play_entropy.get_root())

    print("\n\n")
    print("=== Gini Index ===")

    play_me = id3.ID3("play", gini_index)
    play_me.generate_tree(S, FEATURES)
    play_me.print_tree(play_entropy.get_root())

    # s1 = { "outlook": "S", "temperature": "C", "humidity": "N", "wind": "W", "play":  "+"}
    # print(play_entropy.predict(s1) == s1["play"])

    # s1 = { "outlook": "S", "temperature": "C", "humidity": "N", "wind": "W", "play":  "-"}
    # print(play_entropy.predict(s1) == s1["play"])

    # s1 = { "outlook": "O", "temperature": "C", "humidity": "N", "wind": "W", "play":  "+"}
    # print(play_entropy.predict(s1) == s1["play"])

if __name__ == "__main__":
    main()
