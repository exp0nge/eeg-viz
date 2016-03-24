# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.io import loadmat

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']

# Transpose Matrix to make the rows contain the channels
s = pd.DataFrame(matrix).transpose()

# Create empty 64x64 array
channels_data = [[0 for i in range(64)] for i in range(64)]


# Calculate the correlation between 2 channels
def calculate_correlation(start):
    for index in range(64):
        channels_data[start][index] = pearsonr(s[start], s[index])[0]
    return channels_data[start]

# Create histogram of correlation
for i in range(64):
    print 'doing calculate_correlation for %i' % i
    histogram_data = calculate_correlation(i)
    plt.hist(histogram_data, bins=20, range=(-1, 1))
    plt.title('Channel %s' % i)
    # plt.savefig('Channel %s.svg' % i)
    plt.show()
