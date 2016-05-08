import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import pearsonr,zscore


band1 = np.load('ZScore1.dumps')
band2 = np.load('ZScore2.dumps')
band3 = np.load('ZScore3.dumps')
band4 = np.load('ZScore4.dumps')
band5 = np.load('ZScore5.dumps')
band6 = np.load('ZScore6.dumps')

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
x = np.arange(0,3383.766,.001)

band = np.zeros(shape=(61,3383766))

for index, column in enumerate(band3):
    band[index] = np.concatenate((band1[index],band2[index],band3[index],band4[index],band5[index],band6[index]))

#band = zscore(band)

#matrix=[]
#for index, row in (enumerate(band)):
#	if(index!=63 and index!=62 and index!=61):
#		print index
#		matrix.append(row[3000000:-1])

#matrix = np.array(matrix)

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

#band.dump('ZScore6.dumps')
plt.show()