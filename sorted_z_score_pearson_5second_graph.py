import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy.stats import pearsonr
import cPickle

# load np array
pearson_z_score = np.load('z_score_pearson_5second_dump.dump')
print "Dump loaded"

# absolute value
for i in range(len(pearson_z_score)):
    pearson_z_score[i] = abs(pearson_z_score[i])
print "Obtained Absolute Value"

# sort data
sorted_pearson = sorted(pearson_z_score)
print "Sorted data"

zero = []
for i in range(len(pearson_z_score)):
    zero.append(0)


# plot data

# sorted
plt.plot(range(len(sorted_pearson)), sorted_pearson)
plt.plot(range(len(pearson_z_score)), zero, 'k')


plt.title('Sorted Absolute Value Correlations between 2 Channels (5 second intervals)')
plt.xlim(-1, len(pearson_z_score) + 1)
plt.ylim(0, 1.1)
plt.xlabel('Channel vs Channel')
plt.ylabel('Correlation value')
# plt.xticks(range(64), labels)
plt.savefig('sorted_absolute_value_5second_64_z_score_pearson.svg')
plt.show()

