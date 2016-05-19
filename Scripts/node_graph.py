import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

band1 = np.load('rawData1.dumps').transpose()
band2 = np.load('rawData2.dumps').transpose()
band3 = np.load('rawData3.dumps').transpose()
band4 = np.load('rawData4.dumps').transpose()
band5 = np.load('rawData5.dumps').transpose()
band6 = np.load('rawData6.dumps').transpose()

#band1M = np.load('band1MedCut.dumps').transpose()
#band2M = np.load('band2MedCut.dumps').transpose()
#band3M = np.load('band3MedCut.dumps').transpose()
#band4M = np.load('band4MedCut.dumps').transpose()
#band5M = np.load('band5MedCut.dumps').transpose()
#band6M = np.load('band6MedCut.dumps').transpose()


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
x = np.arange(0,3383.767,.001)

band = np.zeros(shape=(61,3383767))

for index, column in enumerate(band3):
    band[index] = np.concatenate((band1[index],band2[index],band3[index],band4[index],band5[index],band6[index]))

low = 0.3
high = 30
fs = 1000

lowcut = low/(0.5*fs)
highcut = high/(0.5*fs)


#bandpass
b3, a3 = signal.butter(2, [lowcut,highcut], 'band')
w3, h3 = signal.freqs(b3, a3)

matrix=[]
for index, row in (enumerate(band)):
#	if(index!=63 and index!=62 and index!=61):
		print index
		matrix.append(row[3000000:-1])
eeg = signal.detrend(np.array(matrix),type='constant')

mat3 = []
for row in range(len(matrix)):
	mat3.append(signal.lfilter(b3, a3, eeg[row]))

stemp = pd.DataFrame(mat3).clip(-300,300)
#stemp = pd.DataFrame(mat3)

s3 = pd.DataFrame((stemp))
stemp.shape()
band = np.array(s3)


band.dump('bandClipped6.dumps')

#matrix = np.array(matrix)
(s3.transpose()).plot(legend=False)
#for index, column in enumerate(band):
#    #band = band +
#    if(i<=16):
#        ax = fig1.add_subplot(4, 4, i)
#        ax.set_title('Channel %s' % (i))
#        ax.set_xlabel('Sec')
#        ax.plot(x, column)
#    elif(i<=32):
#        ax = fig2.add_subplot(4, 4, i-16)
#        ax.set_title('Channel %s' % (i))
#        ax.set_xlabel('Sec')
#        ax.plot(x, column)
#    elif (i <= 48):
#        ax = fig3.add_subplot(4, 4, i - 32)
#        ax.set_title('Channel %s' % (i))
#        ax.set_xlabel('Sec')
#        ax.plot(x, column)
#    elif (i <= 64):
#        ax = fig4.add_subplot(4, 4, i - 48)
#        ax.set_title('Channel %s' % (i))
#        ax.set_xlabel('Sec')
#        ax.plot(x, column)
#    i+=1

#matrix.dump('MedianSubtract6.dumps')
plt.show()