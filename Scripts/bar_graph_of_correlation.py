# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr

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


# Set up x-axis labels
labels = []
for i in range(64):
    if i % 10 == 0:
        labels.append(i)
    else:
        labels.append('')

# ax = plt.subplot()
# Create histogram of correlation
for i in range(64):
    print 'doing calculate_correlation for %i' % i
    bar_data = calculate_correlation(i)
    # print bar_data
    plt.bar(range(64), bar_data, align='center')

    # Set Title, x and y range, label and add x-axis ticks
    plt.title('Channel %s' % i)
    plt.xlim(-1, 64)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Channels')
    plt.ylabel('Correlations')
    plt.xticks(range(64), labels)
    plt.savefig('Channel %s.svg' % i)
    plt.show()
