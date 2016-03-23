import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
# coding: utf-8
from scipy.io import loadmat

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']
new_matrix = []
for index, row in enumerate(matrix):
	new_matrix.append(row[:5000])
eeg = np.array(new_matrix)
s = pd.DataFrame(new_matrix).transpose()

ssim_data = [[0 for i in range(64)] for i in range(64)]
average = [i for i in range(64)]
variance = [i for i in range(64)]
dyn_range = [i for i in range(64)]


for i in range(len(average)):
    average[i] = np.mean(s[i])
    variance[i] = np.var(s[i])
    dyn_range[i] = np.max(s[i])-np.min(s[i])


# print("Average: ",average)
# print("Variance: ",variance)
# print("Dynamic Range: ",dyn_range)


def ssim(index,indey):
    covxy = np.cov(s[index],s[indey])
    c1 = (0.01 * dyn_range[index])
    c2 = (0.03 * dyn_range[indey])
    ssim_data[index][indey] = (((2*average[index]*average[indey]) + c1)*((2*covxy)+c2))\
                                  /((average[index]**2 + average[indey]**2 + c1)*(variance[index] + variance[indey] + c2))

for i in range(64):
    for j in range(64):
        print("Now doing %d, %d" % (i, j))
        ssim(i,j)

print(ssim_data)
print(ssim_data[0][0])
print(ssim_data[0][1])


'''
def calculate_correlation(start):
	for j in range(4):
		for i in range(16):
			index = 16*j + i
			fig = figs[j]
			ax = fig.add_subplot(4,4,i+1)
			ax.scatter(s[start], s[index])
			coefficients = np.polyfit(s[start],s[index], 1)
			slope, intercept = coefficients
			polynomial = np.poly1d(coefficients)
			ys = polynomial(s[start])
			channels_data[start][index] = pearsonr(s[start], s[index])[0]
			# ax.set_title('s%s vs s%s' %(start, index))
			# ax.plot(s[start],ys)
for i in range(64):
	print('doing calculate_correlation for %i' % i)
	calculate_correlation(i)
print(channels_data[0])
'''

