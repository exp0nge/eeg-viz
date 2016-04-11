import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

m = loadmat('s5d2nap_justdata.mat')

matrix = m['s5d2nap']

s = pd.DataFrame(matrix).transpose()

for i in range(64):
    print "Plotting Channel %s" % i
    plt.plot(range(len(s[i])), s[i])
    plt.title('Channel %s' % i)
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    plt.savefig('Channel %s eeg data.svg' % i)
    plt.show()
