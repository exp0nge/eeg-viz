import matplotlib.pyplot as plt
import numpy as np

# load np array
pearson_z_score = np.load('z_score_pearson_dump.dump')

# absolute value
for i in range(len(pearson_z_score)):
    pearson_z_score[i] = abs(pearson_z_score[i])

# sort data
sorted_pearson = sorted(pearson_z_score)

# plot data
plt.bar(range(len(sorted_pearson)), sorted_pearson)
plt.title('Absolute Value Correlations between 2 Channels')
plt.xlim(-1, 2017)
plt.ylim(0, 1.1)
plt.xlabel('Channel vs Channel')
plt.ylabel('Correlation value')
# plt.xticks(range(64), labels)
plt.savefig('64_absolute_value_z_score_pearson.svg')
plt.show()
