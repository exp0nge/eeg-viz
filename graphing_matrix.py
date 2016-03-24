import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# coding: utf-8

# In[1]:

from scipy.io import loadmat


# In[2]:

m = loadmat('s5d2nap_justdata.mat')


# In[3]:

matrix = m['s5d2nap']

new_matrix = []
for index, row in enumerate(matrix):
	new_matrix.append(row[:5000])
# print len(new_matrix[0])
eeg = np.array(new_matrix)
# In[7]:


# In[8]:

s = pd.DataFrame(new_matrix).transpose()
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
