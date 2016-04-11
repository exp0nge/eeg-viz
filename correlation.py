import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# coding: utf-8

# In[1]:

from scipy.io import loadmat

# In[2]:

m = loadmat('s5d2nap_justdata.mat')

# In[3]:

matrix = m['s5d2nap']

new_matrix = []
for index, row in enumerate(matrix):
    new_matrix.append(row[100000:460000])
# print len(new_matrix[0])
eeg = np.array(new_matrix)
# In[7]:


# In[8]:
s = pd.DataFrame(new_matrix).transpose()
# print s
# plt.subplot(8,8)
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
figs = [fig1, fig2, fig3, fig4]
channels_data = [[0 for i in range(64)] for i in range(64)]


# print len(channels[0]), len(channels)
# start = 0


def calculate_correlation(start):
    for j in range(4):
        for i in range(16):
            index = 16 * j + i
            fig = figs[j]
            ax = fig.add_subplot(4, 4, i + 1)
            ax.scatter(s[start], s[index])
            coefficients = np.polyfit(s[start], s[index], 1)
            slope, intercept = coefficients
            polynomial = np.poly1d(coefficients)
            ys = polynomial(s[start])
            channels_data[start][index] = pearsonr(s[start], s[index])[0]
            ax.set_title('s%s vs s%s' % (start, index))
            ax.plot(s[start], ys, 'r')
            fig.tight_layout()


for i in range(1):
    print 'doing calculate_correlation for %i' % i
    calculate_correlation(i)
print channels_data[0]
# fig.savefig('s0_vs_all.svg')
# fig.tight_layout()
# plt.show()
