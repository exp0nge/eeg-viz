# coding: utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat

MAX_ROW_LENGTH = 10

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']

# The following will take MAX_ROW_LENGTH of each row in our original matrix
new_matrix = []
for ind, row in enumerate(matrix):
    new_matrix.append(row[:MAX_ROW_LENGTH])
eeg = np.array(new_matrix)

s = pd.DataFrame(new_matrix).transpose()

figs = [plt.figure() for i in range(4)]

# Need to store correlation data
channels_data = [[] for each_row in range(64)]
for i in range(64):
    channels_data[i] = range(64)


def calculate_correlation(start):
    for j in range(4):
        index, fig = None, None
        for i in range(16):
            index = 16 * j + i
            fig = figs[j]
            ax = fig.add_subplot(4, 4, i + 1)
            ax.scatter(s[start], s[index])
            coefficients = np.polyfit(s[start], s[index], 1)
            polynomial = np.poly1d(coefficients)
            ys = polynomial(s[start])
            channels_data[start][index] = pearsonr(s[start], s[index])[0]
            ax.set_title('s%s vs s%s' % (start, index))
            ax.plot(s[start], ys)
        fig.tight_layout()
        fig.savefig('corr_%s_v_%s_%i_ts_%i.svg' % (start, index, j, MAX_ROW_LENGTH))


for i in range(1):
    print 'doing calculate_correlation for %i' % i
    calculate_correlation(i)
