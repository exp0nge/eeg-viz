import numpy as np
from scipy.stats import pearsonr
import cPickle

z_scores = np.load('raw_z_npdump.dump')

length_of_intervals = 10000

channels_data = [[[0 for i in range(63)] for i in range(63)] for i in range(len(z_scores[0])/length_of_intervals + 1)]


def calculate_pearson(start):
    for interval in range(len(z_scores[start]) / length_of_intervals + 1):
        j = length_of_intervals * interval
        # print j, interval
        if interval == len(z_scores[start]) / length_of_intervals + 1:
            for index in range(63):
                channels_data[interval][start][index] = pearsonr(z_scores[start][j:], z_scores[index][j:])[0]
        else:
            for index in range(63):
                channels_data[interval][start][index] = pearsonr(z_scores[start][j:j + length_of_intervals],
                                                                 z_scores[index][j:j + length_of_intervals])[0]


def dump_z_score_pearson():
    with open('z_score_pearson_10second_dump.dump', 'wb') as f:
        cPickle.dump(channels_data, f)


if __name__ == '__main__':
    for i in range(63):
        print 'Calculating Channel %s' % i
        calculate_pearson(i)
    dump_z_score_pearson()
