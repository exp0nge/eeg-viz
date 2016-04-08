import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import numpy as np

z_scores = np.load('raw_z_npdump.dump')

for i in range(1):
    print "Plotting Channel %s" % i
    plt.plot(range(len(z_scores[i])), z_scores[i])
    plt.title('Channel %s' % i)
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    plt.savefig('Z_Score Channel %s eeg data.svg' % i)
    plt.show()
