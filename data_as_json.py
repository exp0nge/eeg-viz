# coding: utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.io import loadmat
import json

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']

new_matrix = []
for index, row in enumerate(matrix):
	new_matrix.append(row[:5000])
eeg = np.array(new_matrix)

s = pd.DataFrame(new_matrix).transpose()

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
figs = [fig1, fig2, fig3 ,fig4]
channels_data = {}
for i in range(64):
	channels_data[i] = range(64)
# print channels_data


def calculate_correlation(start):
	for j in range(4):
		for i in range(16):
			index = 16*j + i
			fig = figs[j]
			ax = fig.add_subplot(4,4,i+1)
			ax.scatter(s[start], s[index])
			coefficients = np.polyfit(s[start],s[index], 1)
			slope, intercept = coefficients
			polynomial = np.poly1d(coefficients)
			ys = polynomial(s[start])
			channels_data[start][index] = pearsonr(s[start], s[index])[0]

for i in range(64):
	print 'doing calculate_correlation for %i' % i
	calculate_correlation(i)

with open('correlation_matrix.json', 'w') as outfile:
	json.dump(channels_data, outfile)
