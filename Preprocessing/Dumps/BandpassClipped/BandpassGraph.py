import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import pearsonr,zscore


data1 = np.load('bandClipped1.dumps')
#data1.dump('band1MedCut.dumps')
data2 = np.load('bandClipped2.dumps')
#data2.dump('band2MedCut.dumps')
data3 = np.load('bandClipped3.dumps')
#data3.dump('band3MedCut.dumps')
data4 = np.load('bandClipped4.dumps')
#data4.dump('band4MedCut.dumps')
data5 = np.load('bandClipped5.dumps')
#data5.dump('band5MedCut.dumps')
data6 = np.load('bandClipped6.dumps')
#data6.dump('band6MedCut.dumps')

#data1 = np.load('FMedian1.dumps')
#data2 = np.load('FMedian2.dumps')
#data3 = np.load('FMedian3.dumps')
#data4 = np.load('FMedian4.dumps')
#data5 = np.load('FMedian5.dumps')
#data6 = np.load('FMedian6.dumps')


fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
figs = [fig1, fig2, fig3, fig4]
channels_data = [[0 for i in range(63)] for i in range(63)]

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
    if (index != 63 and index != 62 and index != 61):
	    data[index] = np.concatenate((data1[index],data2[index],data3[index],data4[index],data5[index],data6[index]))

#data1 = signal.medfilt(data1)
#data2 = signal.medfilt(data2)
#data3 = signal.medfilt(data3)
#data4 = signal.medfilt(data4)
#data5 = signal.medfilt(data5)
#data6 = signal.medfilt(data6)

#data = signal.medfilt(data)
#data = zscore(data)
print "So...."

#matrix=[]
#for index, row in (enumerate(data)):
#	if(index!=63 and index!=62 and index!=61):
#		print index
#		matrix.append(row[1800000:2400000])
 #       print "Is it working now?"

#matrix = np.array(matrix)

#print "So...."
for index, column in enumerate(data):
    #band = band +
    if(i<=16):
        ax = fig1.add_subplot(4, 4, i)
        ax.set_title('Channel %s' % (i))
        ax.set_xlabel('Sec')
        ax.plot(x, column)
    elif(i<=32):
        ax = fig2.add_subplot(4, 4, i - 16)
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
        print "Is it "
    i+=1

#data1.dump('Median1.dumps')
#data2.dump('Median2.dumps')
#data3.dump('Median3.dumps')
#data4.dump('Median4.dumps')
#data5.dump('Median5.dumps')
#data6.dump('Median6.dumps')

#matrix.dump('FMedian4.dumps')


print "Well?"
plt.show()