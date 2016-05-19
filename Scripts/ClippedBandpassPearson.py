import matplotlib.pyplot as plt
import numpy as np
import simplejson
from scipy.stats import pearsonr,zscore
from sklearn.cluster import SpectralBiclustering

band1 = np.load('band1.dumps').transpose()

band1 = np.load('band1MedCut.dumps').transpose()
band2 = np.load('band2MedCut.dumps').transpose()
band3 = np.load('band3MedCut.dumps').transpose()
band4 = np.load('band4MedCut.dumps').transpose()
band5 = np.load('band5MedCut.dumps').transpose()
band6 = np.load('band6MedCut.dumps').transpose()

band = np.zeros(shape=(63,3383767))

for index, column in enumerate(band3):
    band[index] = np.concatenate((band1[index],band2[index],band3[index],band4[index],band5[index],band6[index]))

band = zscore(np.array(band))

length_of_intervals = 30000
channels = range(63)
iterations = range(len(band[0])/length_of_intervals)
pearson_data = [[] for j in iterations]

# def pearson_(interval):
#     print '|',
#     start, end = interval, interval + length_of_intervals
#     for i in channels:
#         for j in range(63):
#             if i <= j:
#                 pearson_data[interval].append(pearsonr(data[i][start:end], data[j][start:end])[0])
start, end = 0,  length_of_intervals
interval = 0

while end <= len(band[0]):
    print interval,
    for i in channels:
        for j in range(63):
            if i <= j:
                pearson_data[interval].append(pearsonr(band[i][start:end], band[j][start:end])[0])
    start = end
    end += length_of_intervals
    interval += 1

p = np.array(pearson_data)
spectral_model = SpectralBiclustering()
spectral_model.fit(p)


fit_data = p[np.argsort(spectral_model.row_labels_)]
fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
plt.matshow(p, cmap=plt.cm.Blues)
plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.matshow(np.outer(np.sort(spectral_model.row_labels_) + 1,
                     np.sort(spectral_model.column_labels_) + 1),
            cmap=plt.cm.Blues)

with open('media/pearson_30sec_bandpassMedian_clipped_2016.json', 'w+') as f:
    pearson_data_r = np.array(pearson_data)
    p = [[float(column) for column in row] for row in pearson_data_r]
    f.write(simplejson.dumps({'name': 's5d2nap', 'data': p}))
plt.show()
