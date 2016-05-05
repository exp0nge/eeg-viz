import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
band1 = np.load('band1.dumps').transpose()

band1 = np.load('band1cut.dumps').transpose()
band2 = np.load('band2cut.dumps').transpose()
band3 = np.load('band3cut.dumps').transpose()
band4 = np.load('band4cut.dumps').transpose()
band5 = np.load('band5cut.dumps').transpose()
band6 = np.load('band6cut.dumps').transpose()

band1M = np.load('band1MedCut.dumps').transpose()
band2M = np.load('band2MedCut.dumps').transpose()
band3M = np.load('band3MedCut.dumps').transpose()
band4M = np.load('band4MedCut.dumps').transpose()
band5M = np.load('band5MedCut.dumps').transpose()
band6M = np.load('band6MedCut.dumps').transpose()


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

band = np.zeros(shape=(63,3383767))

for index, column in enumerate(band3):
    band[index] = np.concatenate((band1[index]-band1M[index],band2[index]-band2M[index],band3[index]-band3M[index],band4[index]-band4M[index],band5[index]-band5M[index],band6[index]-band6M[index]))


for index, column in enumerate(band):
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
#bandpass = np.array(band.transpose())
#bandpass.dump('bandcut3.dumps')
plt.show()