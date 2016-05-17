import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from scipy.io import loadmat
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering

pearson_band = np.load('band_10s_pearson.dump')
print "Dump loaded"
abs_val = []
zero = []
# absolute value and create 0 line
for i in range(len(pearson_band)):
    abs_val.append(abs(pearson_band[i]))
    zero.append(0)
print "Obtained Absolute Value"

# sort data
sorted_pearson = sorted(pearson_band)
sorted_abs_val = sorted(abs_val)
print "Sorted data"

# plot data

# sorted no abs val
plt.plot(range(len(sorted_pearson)), sorted_pearson)
plt.plot(range(len(pearson_band)), zero, 'k')

plt.title('Sorted Correlations between 2 Channels (10 second intervals)')
plt.xlim(-1, len(sorted_pearson) + 1)
plt.ylim(-1.1, 1.1)
plt.xlabel('Channel vs Channel')
plt.ylabel('Correlation value')
# plt.xticks(range(64), labels)
plt.savefig('sorted_band_10s_pearson.svg')
plt.show()


# sorted abs val
plt.plot(range(len(sorted_abs_val)), sorted_abs_val)
plt.title('Sorted Absolute Value Correlations between 2 Channels (10 second intervals)')
plt.xlim(-1, len(sorted_pearson) + 1)
plt.ylim(0, 1.1)
plt.xlabel('Channel vs Channel')
plt.ylabel('Correlation value')
# plt.xticks(range(64), labels)
plt.savefig('sorted_absolute_value_10s_full_pearson.svg')
plt.show()
