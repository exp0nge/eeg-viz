import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import pearsonr,zscore
from sklearn import preprocessing


data1 = np.load('rawData1.dumps')
data2 = np.load('rawData2.dumps')
data3 = np.load('rawData3.dumps')
data4 = np.load('rawData4.dumps')
data5 = np.load('rawData5.dumps')
data6 = np.load('rawData6.dumps')

#data1M = np.load('band1MedCut.dumps').transpose()
#data2M = np.load('band2MedCut.dumps').transpose()
#data3M = np.load('band3MedCut.dumps').transpose()
#data4M = np.load('band4MedCut.dumps').transpose()
#data5M = np.load('band5MedCut.dumps').transpose()
#data6M = np.load('band6MedCut.dumps').transpose()


fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
figs = [fig1, fig2, fig3, fig4]
channels_data = [[0 for i in range(64)] for i in range(64)]

#def calculate_correlation(start):
#    for j in range(4):
#        for i in range(16):
#            index = 16 * j + i
#            fig = figs[j]
#            ax = fig.add_subplot(4, 4, i + 1)
#            channels_data[start][index] = pearsonr(s[start], s[index])[0]
#            ax.plot(s[start], ys, 'r')
#            fig.tight_layout()

i=1
x = np.arange(0,3383.766,.001)

data = np.zeros(shape=(61,3383766))

for index, column in enumerate(data3):
	data[index] = np.concatenate((data1[index],data2[index],data3[index],data4[index],data5[index],data6[index]))

data = preprocessing.robust_scale(data,True,True,True)
#eeg = signal.detrend(np.array(data),type='constant')
#data = signal.medfilt(eeg)

#data = eeg-data

#matrix=[]
#for index, row in (enumerate(data)):
#	if(index!=63 and index!=62 and index!=61):
#		print index
#		matrix.append(row[3000000:-1])

#matrix = np.array(matrix)

for index, column in enumerate(data):
    #band = band +
    if(i<=16):
        ax = fig1.add_subplot(4, 4, i)
        ax.set_title('Channel %s' % (i))
        ax.set_xlabel('Sec')
        ax.plot(x, column)
    elif(i<=32):
        ax = fig2.add_subplot(4, 4, i-16)
        ax.set_title('Channel %s' % (i))
        ax.set_xlabel('Sec')
        ax.plot(x, column)
    elif (i <= 48):
        ax = fig3.add_subplot(4, 4, i - 32)
        ax.set_title('Channel %s' % (i))
        ax.set_xlabel('Sec')
        ax.plot(x, column)
    elif (i <= 64):
        ax = fig4.add_subplot(4, 4, i - 48)
        ax.set_title('Channel %s' % (i))
        ax.set_xlabel('Sec')
        ax.plot(x, column)
    i+=1

#data.dump('Median6.dumps')
plt.show()