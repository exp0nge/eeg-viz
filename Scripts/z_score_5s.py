import cPickle

import numpy as np
from scipy import stats
from scipy.io import loadmat

m = loadmat('s5d2nap_justdata.mat')
matrix = np.array(m['s5d2nap'])

# Z Scores 5 second intervals
z_score_five_sec = []


def five_second_z_score(start):
    for interval in range(len(matrix[start])/5000 + 1):
        j = 5000*interval
        # print interval, j
        if interval == len(matrix[start])/5000 + 1:
            z_score_five_sec.append(stats.zscore(matrix[start][j:]))
        else:
            z_score_five_sec.append(stats.zscore(matrix[start][j:j+5000]))


def dump_raw_z_scores():
    with open('raw_z_npdump_5s_interval.dump', 'wb') as f:
        cPickle.dump(z_score_five_sec, f)

for i in range(64):
    print "Channel %s" % i
    five_second_z_score(i)

dump_raw_z_scores()
