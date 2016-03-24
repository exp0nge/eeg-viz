import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# coding: utf-8

# In[1]:

from scipy.io import loadmat
from scipy import signal

# In[2]:

m = loadmat('s5d2nap_justdata.mat')


# In[3]:

matrix = m['s5d2nap']

b, a = signal.butter(2, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
new_matrix = []
for index, row in enumerate(matrix):
	new_matrix.append(row[2000:5000])
# print len(new_matrix[0])
eeg = np.array(new_matrix)
# In[7]:

# In[8]:
y = signal.lfilter(b, a, eeg[0])
print y

mat = []
for row in range(len(new_matrix)):
	mat.append(signal.lfilter(b, a, eeg[row]))

s = pd.DataFrame(mat).transpose()
print s


# In[ ]:
s.plot()
# cumsum() adds the value of each channel and displays the sum
# s = s.cumsum()
# s.plot()

# fig = plt.figure()
# fig.savefig('matrix.svg')


plt.figure(2);

# plt.plot(eeg[:8,:30000].T + 8000*np.arange(7,-1,-1));

# plt.plot(np.zeros((30000,8)) + 8000*np.arange(7,-1,-1),'--',color='gray');

plt.yticks([]);

# plt.legend(first['channels']);

plt.axis('tight');


plt.show()
