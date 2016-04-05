# coding: utf-8
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
from scipy.io import loadmat
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering

import cPickle

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']
print 'Data loaded'


def calculate_pearson_correlation(data_matrix, store_matrix, start):
    # Need to store correlation data
    s = pd.DataFrame(data_matrix)
    s = s.transpose()
    figs = [plt.figure() for i in range(4)]
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
            store_matrix[start][index] = pearsonr(s[start], s[index])[0]
            # Follow is to dump graphs as SVG
            # ax.set_title('s%s vs s%s' % (start, index))
            # ax.plot(s[start], ys)
            # fig.tight_layout()
            # fig.savefig('corr_%s_v_%s_%i_ts_%i.svg' % (start, index, j, MAX_ROW_LENGTH))
    return store_matrix


def calculate_n_columns(data_matrix, n=64):
    channels_data = [[] for each_row in range(64)]
    for row in range(64):
        channels_data[row] = range(64)

    for row in range(n):
        print 'doing calculate_correlation for %i' % row
        channels_data = calculate_pearson_correlation(data_matrix, channels_data, row)
    return data_matrix


def slice_matrix(data_matrix, first_n_elements):
    channels_data = [[] for each_row in range(64)]
    for row in range(64):
        channels_data[row] = data_matrix[row][:first_n_elements]

    return np.array(channels_data)


def plot_biclustering_with_pearson(time_ms, title):
    sliced_matrix = slice_matrix(matrix, time_ms)
    channels_data = calculate_n_columns(sliced_matrix)
    z_score = stats.zscore(channels_data)
    plt.title('Z Score Biclustering Over %i ms' % time_ms)
    spectral_model = SpectralBiclustering()
    spectral_model.fit(z_score)
    fit_data = z_score[np.argsort(spectral_model.row_labels_)]
    fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.savefig('z_score_%s_biclustering_all_ts_%i.svg' % (time_ms, title))


def plot_biclustering_raw_data(time_ms, t=False):
    # take the transpose of sliced matrix
    if t:
        channels_data = slice_matrix(matrix, time_ms).T
    else:
        channels_data = slice_matrix(matrix, time_ms)
    print len(channels_data), len(channels_data[1])
    z_score = stats.zscore(channels_data)
    plt.title('Z Score Biclustering Over %i ms' % time_ms)
    spectral_model = SpectralBiclustering()
    spectral_model.fit(z_score)
    fit_data = z_score[np.argsort(spectral_model.row_labels_)]
    fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    # plt.savefig('z_score_raw_biclustering_all_ts_%i_T_%s.svg' % (time_ms, str(t)))
    plt.show()


def plot_coclusters_raw_data(time_ms, t=False):
    # take the transpose of sliced matrix
    if t:
        channels_data = slice_matrix(matrix, time_ms)
    else:
        channels_data = slice_matrix(matrix, time_ms)
    print len(channels_data), len(channels_data[1])
    z_score = stats.zscore(channels_data)
    plt.title('Z Score Biclustering Over %i ms' % time_ms)
    spectral_model = SpectralCoclustering()
    spectral_model.fit(z_score)
    fit_data = z_score[np.argsort(spectral_model.row_labels_)]
    fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.savefig('z_score_raw_coclustering_all_ts_%i_T_%s.svg' % (time_ms, str(t)))


def plot_biclusters_n_intervals(n_intervals=30000):
    channels_data = [[] for i in range(64)]
    for row in range(64):
        start, end = 0, n_intervals
        row_data = matrix[row]
        while end < len(row_data):
            channels_data[row].append(float(sum(row_data[start:end])) / len(row_data[start:end]))
            start = end
            end += n_intervals
    z_score = stats.zscore(np.array(channels_data))
    plt.title('Z Score Biclustering')
    spectral_model = SpectralBiclustering()
    spectral_model.fit(z_score)
    fit_data = z_score[np.argsort(spectral_model.row_labels_)]
    fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.savefig('z_score_raw_biclustering_all_%is.svg' % (n_intervals / 1000))


def dump_raw_z_scores():
    z_score = stats.zscore(np.array(matrix))
    with open('raw_z_scores_array', 'wb') as f:
        cPickle.dump(z_score, f)


if __name__ == '__main__':
    # plot_biclustering_with_pearson(30000000000)
    # plot_biclustering_raw_data(60000)
    # plot_biclustering_raw_data(60000, t=True)
    # plot_coclusters_raw_data(60000)
    # plot_coclusters_raw_data(60000, t=True)
    # plot_biclusters_n_intervals(15000)
    dump_raw_z_scores()
