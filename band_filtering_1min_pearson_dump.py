import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
# from scipy.io import loadmat
# from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering

import cPickle

band1 = np.load('band1.dump')
band2 = np.load('band2.dump')
band3 = np.load('band3.dump')
band4 = np.load('band4.dump')
band5 = np.load('band5.dump')

band = np.transpose(np.concatenate((band1, band2, band3, band4, band5)))
print band.shape
pearson_5s = []
length = 60000


def calculate_pearson(start):
    for interval in range(len(band[start])/length + 1):
        j = length*interval
        # print j, interval
        if interval == len(band[start])/length + 1:
            for index in range(63):
                if start < index:
                    pearson_5s.append(pearsonr(band[start][j:], band[index][j:])[0])
        else:
            for index in range(63):
                if start < index:
                    pearson_5s.append(pearsonr(band[start][j:j+length], band[index][j:j+length])[0])


def dump_band_pearson():
    with open('band_1min_pearson.dump', 'wb') as f:
        cPickle.dump(pearson_5s, f)


if __name__ == '__main__':
    for i in range(63):
        print 'Calculating Channel %s' % i
        calculate_pearson(i)
    dump_band_pearson()
