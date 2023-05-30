#!/usr/bin/env python

import sys
import csv
import numpy
import seaborn
import pandas
from matplotlib import pylab


if __name__ == "__main__":

    table_path = sys.argv[1]

    data = pandas.read_csv(table_path, sep=',', index_col=0, header=0)

    heatmap = seaborn.heatmap(data, cmap="Greys_r")

    pylab.title(table_path)

    pylab.show()
