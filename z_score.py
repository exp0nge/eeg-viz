# coding: utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat

m = loadmat('s5d2nap_justdata.mat')
MAX_ROW_LENGTH = 1

matrix = m['s5d2nap']

eeg = np.array(matrix[:MAX_ROW_LENGTH])

s = pd.DataFrame(matrix).transpose()

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
figs = [fig1, fig2, fig3, fig4]
channels_data = {}
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
            ax.set_title('s%s vs s%s, j: %i' % (start, index, j))
            ax.plot(s[start], ys)
        fig.tight_layout()
        fig.savefig('%s_v_%s_%i.svg' % (start, index, j))


for i in range(1):
    print 'doing calculate_correlation for %i' % i
    calculate_correlation(i)
