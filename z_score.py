# coding: utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
from scipy.io import loadmat
from sklearn.cluster.bicluster import SpectralBiclustering
import cPickle

MAX_ROW_LENGTH = 3000000
CHANNELS = 1

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']

# The following will take MAX_ROW_LENGTH of each row in our original matrix
# new_matrix = []
# for ind, row in enumerate(matrix):
#     new_matrix.append(row[:MAX_ROW_LENGTH])
# eeg = np.array(new_matrix)

s = pd.DataFrame(matrix).transpose()

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
            # fig = figs[j]
            # ax = fig.add_subplot(4, 4, i + 1)
            # ax.scatter(s[start], s[index])
            coefficients = np.polyfit(s[start], s[index], 1)
            polynomial = np.poly1d(coefficients)
            ys = polynomial(s[start])
            channels_data[start][index] = pearsonr(s[start], s[index])[0]
            # Follow is to dump graphs as SVG
            # ax.set_title('s%s vs s%s' % (start, index))
            # ax.plot(s[start], ys)
        # fig.tight_layout()
        # fig.savefig('corr_%s_v_%s_%i_ts_%i.svg' % (start, index, j, MAX_ROW_LENGTH))


for i in range(64):
    print 'doing calculate_correlation for %i' % i
    calculate_correlation(i)

# z_score = stats.zscore(channels_data)
# plt.title('Z Score Biclustering Over %i ms' % MAX_ROW_LENGTH)
# spectral_model = SpectralBiclustering()
# spectral_model.fit(z_score)
# fit_data = z_score[np.argsort(spectral_model.row_labels_)]
# fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
# plt.matshow(fit_data, cmap=plt.cm.Blues)
# plt.savefig('z_score_biclustering_%i_vs_all_ts_%i.svg' % (0, MAX_ROW_LENGTH))
# plt.show()

with open('pearson_r_all_64_dictionary_dump', 'wb') as f:
    cPickle.dump(channels_data, f)
