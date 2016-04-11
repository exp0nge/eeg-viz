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

#lowpass
b1, a1 = signal.butter(2, 0.1, 'low', analog=True)
w1, h1 = signal.freqs(b1, a1)

#highpass
b2, a2 = signal.butter(2, 0.5, 'high', analog=True)
w2, h2 = signal.freqs(b2, a2)

low = 0.3
high = 30
fs = 1000

lowcut = low/(0.5*fs)
highcut = high/(0.5*fs)


#bandpass
b3, a3 = signal.butter(2, [lowcut,highcut], 'band')
w3, h3 = signal.freqs(b3, a3)

#create matrix
new_matrix = []

for index, row in (enumerate(matrix)):
	if(index!=63):
		new_matrix.append(row[2400000:-1])
# print len(new_matrix[0])
eeg = signal.detrend(np.array(new_matrix),type='constant')


#normal print
#]mat = []
#for row in range(len(new_matrix)):
#	mat.append(eeg[row])

#s = pd.DataFrame(mat).transpose()

#lowpass print
#mat1 = []
#for row in range(len(new_matrix)):
#	mat1.append(signal.lfilter(b1, a1, eeg[row]))

#s1 = pd.DataFrame(mat1).transpose()

#highpass print
#mat2 = []
#for row in range(len(new_matrix)):
#	mat2.append(signal.lfilter(b2, a2, eeg[row]))
#s2 = pd.DataFrame(mat2).transpose()

#bandpass
mat3 = []
for row in range(len(new_matrix)):
	mat3.append(signal.lfilter(b3, a3, eeg[row]))

s3 = pd.DataFrame(mat3).transpose()

band = np.array(s3)
band.dump('band5.dumps')

# In[ ]:
#s.plot(legend=False)
#s1.plot(legend=False)
#s2.plot(legend=False)
s3.plot(legend=False)

# cumsum() adds the value of each channel and displays the sum
# s = s.cumsum()
# s.plot()


# fig = plt.figure()
# fig.savefig('matrix.svg')


#plt.figure(2);

# plt.plot(eeg[:8,:30000].T + 8000*np.arange(7,-1,-1));

# plt.plot(np.zeros((30000,8)) + 8000*np.arange(7,-1,-1),'--',color='gray');

#plt.yticks([]);

# plt.legend(first['channels']);

#plt.axis('tight');


plt.show()
