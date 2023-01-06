import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from tools.features import norm_1_sig

def highlightSubsequences(X, scores, k, win_size):
    #from the retrieved scores, display the segments found put scores in same dimension as SIGNAL
    new_scores = np.zeros(len(scores) + win_size)
    new_scores[int(win_size / 2): -int(win_size/2)] = scores
    x_scores = np.linspace(1, len(new_scores), len(new_scores))
    half_win = int(win_size / 2)

    #3 - plot score function on axes
    plt.plot(new_scores)

    #4 Order by k-most representative windows
    #5 highlight most important windows
    #6 plot eveything in the corresponding plots
    arg_k = np.zeros(k)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    if (np.ndim(X) > 1):
        for i in range(np.shape(X)[1]):
            ax1.plot(norm_1_sig(X[:, 1])+(i - 1) * 1.5, linewidth=1, color='#808080')
    else:
        ax1.plot(norm_1_sig(X), linewidth=0.8, color='#808080')

    cmap = matplotlib.cm.get_cmap('jet')

    for k_i in range(k):
        # find max
        argmax_ki = np.argmax(new_scores)
        max_ki = new_scores[argmax_ki]
        # save index to plot around it
        arg_k[k_i] = argmax_ki

        #remove the score and its surrounding
        a = argmax_ki - win_size//2
        b = argmax_ki + win_size//2

        new_scores[a:b] = 0
        # highlight on plot
        if (np.ndim(X) > 1):
            for i in range(np.shape(X)[1]):
                #plot subsequences on signal
                s_ = norm_1_sig(X[:, i])
                ax2.plot(x_scores[a: b], s_[a: b]+(i - 1) * 1.5, color=cmap(int(max_ki)), linewidth=2)
                # plot ordered on side
                x_i = np.linspace(-1 + i, i - 0.5, b-a)
                ax2.plot(x_i, s_[a: b]+(k - k_i) * max_ki, linewidth=2, color=cmap(int(max_ki)))
        else:
            s_ = norm_1_sig(X)
            ax1.plot(x_scores[a:b], s_[a:b], color=cmap(max_ki), linewidth=2)
            x_i = np.linspace(0, 0.5, b-a)
            ax2.plot(x_i, s_[a:b]+(k - k_i), linewidth=2)

    plt.show()