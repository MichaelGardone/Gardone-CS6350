LABELS = ["+", "-"]

COLUMNS = ["outlook", "temperature", "humidity", "wind", "play"]

FEATURES = { "outlook":      ['S', 'O', 'R'],
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