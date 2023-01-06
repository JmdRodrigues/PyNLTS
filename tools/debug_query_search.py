import numpy as np
from word_feature_vectors import word_feature_extraction, multi_dimensional_extraction
from features import norm_1_sig, movmean
from query_search import query_search
from plot_tools import highlightSubsequences
import matplotlib.pyplot as plt

s = np.loadtxt("../data/opensignals_ANDROID_ACCELEROMETER_2022-02-12_11-29-16.txt")[:50000, 2]
# s = s[:, 0]
win_size = 1500
#% 1 - extract word feature vectors
key1, key2, key3 = word_feature_extraction(s, win_size)
# key1, key2, key3 = word_feature_extraction(s, win_size)
query_input = "s1: [down down up] middle"
query_input2 = "s1: !high"
query_input3 = "s1: low"
k = 10
scores = query_search(s, key1, key2, key3, [query_input], win_size)
highlightSubsequences(s, scores, int(k), int(win_size))

scores2 = query_search(s, key1, key2, key3, [query_input2], win_size)
scores3 = query_search(s, key1, key2, key3, [query_input3], win_size)

plt.plot(s)

plt.plot(scores, 'k')
plt.plot(scores2, 'g')
plt.plot(scores3, 'y')
plt.show()