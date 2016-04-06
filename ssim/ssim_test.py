import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from scipy.stats import pearsonr
from scipy import stats
from sklearn.cluster.bicluster import SpectralBiclustering
from scipy.io import loadmat
from random import randint

m = loadmat('s5d2nap_justdata.mat')

# length of time
offset = 500

# random range
matrix_range = 5000 #random.randint(0, len(matrix[0]) - 500)

matrix = m['s5d2nap']
new_matrix = []
print("Len of matrix[0] = %d" % len(matrix[0]))
print("Range: %d , %d" % (matrix_range, matrix_range + offset))
for index, row in enumerate(matrix):
    new_matrix.append(row[matrix_range:matrix_range + offset])
eeg = np.array(new_matrix)
s = pd.DataFrame(matrix).transpose()

# min_num = np.min(np.min(new_matrix))
# print("Min: %f " % min_num)
# max_num = np.max(np.max(new_matrix))
# print("Max: %f" % max_num)
# range_num = max_num - min_num
# print("Range: %f" % range_num)
#
# threshold_range = []
#
# for i in range((int)(math.floor(min_num)), (int)(math.floor(max_num)), (int)(math.floor((range_num / 25)))):
#     threshold_range.append(i)
# threshold_range[0] = math.floor(min_num)
# threshold_range[len(threshold_range) - 1] = math.ceil(max_num)
# print(threshold_range)

# threshold_array = [0 for i in range(len(threshold_range))]

# for i in range(0, len(new_matrix) - 1):
#     for j in range(0, len(new_matrix[0]) - 1):
#         # print("%d, %d" %(i,j))
#         for k in range(0, len(threshold_range) - 1):
#             if (new_matrix[i][j] > threshold_range[k] and new_matrix[i][j] < threshold_range[k + 1]):
#                 threshold_array[k] += 1
#                 k = len(threshold_range) - 2
# print("Threshold Array Done")
# print(threshold_array)



# with open("ssim_range", "wb") as f:
#     pickle.dump(s_range, f)
#
# with open("ssim_range", "rb") as f:
#     s_range = pickle.load(f)

# ssim_data = [[0 for i in range(64)] for i in range(64)]
# average = [i for i in range(64)]
# variance = [i for i in range(64)]
# dyn_range = [i for i in range(64)]
#
# for i in range(len(average)):
#     average[i] = np.mean(new_matrix[i])
#     variance[i] = np.var(new_matrix[i])
#     dyn_range[i] = np.max(new_matrix[i]) - np.min(new_matrix[i])


# Debug
# print("Average: ",average)
# print("Variance: ",variance)
# print("Dynamic Range: ",dyn_range)


# def ssim(index, indey):
#     covxy = np.cov(s[index], s[indey])[0][1]
#     c1 = (0.01 * dyn_range[index])
#     c2 = (0.03 * dyn_range[indey])
#     ssim_data[index][indey] = (((2 * average[index] * average[indey]) + c1) * ((2 * covxy) + c2)) \
#                               / ((average[index] ** 2 + average[indey] ** 2 + c1) * (variance[index] + variance[indey] + c2))

# for i in range(64):
#     for j in range(64):
#         print("SSIM %d, %d" % (i, j))
#         ssim(i, j)
# print("SSIM Complete")

# Reads SSIM_data
# with open("ssim_data", "rb") as f:
#     ssim_data = pickle.load(f)



# for i in range(len(ssim_data)):
#     for j in range(len(ssim_data[0])):
#         biclusterdata.append(ssim_data[i][j])


# print(len(biclusterdata))

# Writes to SSIM_data
# with open("ssim_data", "wb") as f:
#     pickle.dump(ssim_data, f)

# z_score = stats.zscore(ssim_data)
# spectral_model = SpectralBiclustering()
# spectral_model.fit(z_score)
# fit_data = z_score[np.argsort(spectral_model.row_labels_)]
# fit_data = fit_data[:, np.argsort(spectral_model.column_labels_)]
# plt.matshow(fit_data, cmap=plt.cm.Blues)
# plt.title('SSIM from %d to %d' % (matrix_range, matrix_range + offset))
#
# plt.savefig('ssim_z_score.svg')



# plt.bar(threshold_range, threshold_array, 100, color="blue")
# plt.title('Thresholds from %d to %d' % (matrix_range, matrix_range + offset))

# plt.show()
