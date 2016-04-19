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
full_pearson = []


def calculate_pearson(start):
    for index in range(63):
        if start < index:
            full_pearson.append(pearsonr(band[start], band[index])[0])


def dump_z_score_pearson():
    with open('band_full_pearson.dump', 'wb') as f:
        cPickle.dump(full_pearson, f)


if __name__ == '__main__':
    for i in range(63):
        print 'Calculating Channel %s' % i
        calculate_pearson(i)
    dump_z_score_pearson()
