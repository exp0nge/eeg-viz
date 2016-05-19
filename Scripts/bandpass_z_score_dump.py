import cPickle

import numpy as np
from scipy import stats

band1 = np.load('band1.dump')
band2 = np.load('band2.dump')
band3 = np.load('band3.dump')
band4 = np.load('band4.dump')
band5 = np.load('band5.dump')
print "loaded"

band = np.transpose(np.concatenate((band1, band2, band3, band4, band5)))
print "combined"
z_scores = stats.zscore(np.array(band))
print "calculated z_scores"
print len(z_scores), len(z_scores[0])
def dump_band_z_scores():
    with open('bandpass_z_score_dump.dump', 'wb') as f:
        cPickle.dump(z_scores, f)


dump_band_z_scores()