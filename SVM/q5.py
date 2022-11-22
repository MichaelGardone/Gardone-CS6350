# python/native OS
import os

# 3rd party
import numpy

import PrimalSVM

def main():
    tr_x = [[0.5, -1, 0.3, 1], [-1, -2, -2, 1], [1.5, 0.2, -2.5, 1]]
    tr_y = [1, -1, 1]

    psvm = PrimalSVM.PrimalSVM()
    psvm.create_verbose_classifier(numpy.array(tr_x), numpy.array(tr_y), 1/3)

    return 0

###

if __name__ == "__main__":
    main()