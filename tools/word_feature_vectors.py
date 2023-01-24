import numpy as np
from tools.features import *

def multi_scale_vector_generation(signal, win_size):
    noise_i, complex_i, simple_i, clean_i, high_i, low_i, bottom_i, top_i, mean_i, up_i, down_i, speed_i, flat_i = quick_moving_feature_extraction(signal, win_size)
    symmetry_i = slow_moving_feature_extraction(signal, win_size)
    # symmetry_i = norm_1_sig(np.random.rand(len(signal)-win_size+1))
    assymetry_i = 1-symmetry_i

    peak_i = gauss_peak_moving(signal, 0.6, win_size)
    valley_i = gauss_val_moving(signal, 0.6, win_size)

    vval_i = v_gauss_val_moving(signal, win_size)
    uval_i = u_gauss_val_moving(signal, win_size)

    pos_step_i, neg_step_i = moving_step(signal, win_size)
    pos_plat_i, neg_plat_i = moving_plateau(signal, win_size)
    # [uncommon_i, common_i] = MP([fliplr(app.SIGNAL(1:win_size / 2)), app.SIGNAL, fliplr(app.SIGNAL(end - win_size / 2: end))], win_size)

    keywords_vals = [noise_i, up_i, down_i, flat_i, symmetry_i, assymetry_i, complex_i, high_i, low_i, peak_i, valley_i, pos_step_i,
                     neg_step_i, pos_plat_i, neg_plat_i, top_i, bottom_i, simple_i, speed_i, vval_i, uval_i, mean_i, clean_i]

    return keywords_vals

def multi_d_scale_vector_generation(X, win_size, keyword_keys):
    print(np.shape(X))
    d = {str(i+1): [] for i in range(np.shape(X)[1])}
    for i, signal_i in enumerate(np.transpose(X)):
        noise_i, complex_i, simple_i, clean_i, high_i, low_i, bottom_i, top_i, mean_i, up_i, down_i, speed_i, flat_i = quick_moving_feature_extraction(signal_i, win_size)
        symmetry_i = slow_moving_feature_extraction(signal_i, win_size)
        # symmetry_i = norm_1_sig(np.random.rand(len(signal)-win_size+1))
        assymetry_i = 1-symmetry_i

        peak_i = gauss_peak_moving(signal_i, 0.6, win_size)
        valley_i = gauss_val_moving(signal_i, 0.6, win_size)

        vval_i = v_gauss_val_moving(signal_i, win_size)
        uval_i = u_gauss_val_moving(signal_i, win_size)

        pos_step_i, neg_step_i = moving_step(signal_i, win_size)
        pos_plat_i, neg_plat_i = moving_plateau(signal_i, win_size)
        # [uncommon_i, common_i] = MP([fliplr(app.SIGNAL(1:win_size / 2)), app.SIGNAL, fliplr(app.SIGNAL(end - win_size / 2: end))], win_size)

        keywords_vals = [noise_i, up_i, down_i, flat_i, symmetry_i, assymetry_i, complex_i, high_i, low_i, peak_i, valley_i, pos_step_i,
                         neg_step_i, pos_plat_i, neg_plat_i, top_i, bottom_i, simple_i, speed_i, vval_i, uval_i, mean_i, clean_i]

        d[str(i+1)] = dict(zip(keyword_keys, keywords_vals))


    return d

def uni_dimensional_extraction(signal, win_size):
    # map keywords and keys into a container, already normalized
    keywords_keys = ['noise', 'up', 'down', 'flat', 'symmetric', 'assymetric', 'complex', 'high', 'low', 'peak', 'valley', 'step_up',
                     'step_down', 'plateau_up', 'plateau_down', 'top', 'bottom', 'simple', 'quick', 'vval', 'uval',
                     'middle', 'clean']

    keywords_vals1 = multi_scale_vector_generation(signal, int(win_size))
    keywords_vals2 = multi_scale_vector_generation(signal, int(win_size/2))
    # window2 = win_size / 16
    if (win_size / 4 < 2):
        win_size = 32
    keywords_vals3 = multi_scale_vector_generation(signal, int(win_size / 4))

    KEY_VECTOR1 = dict(zip(keywords_keys, keywords_vals1))
    KEY_VECTOR2 = dict(zip(keywords_keys, keywords_vals2))
    KEY_VECTOR3 = dict(zip(keywords_keys, keywords_vals3))

    return KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3

def multi_dimensional_extraction(X, win_size):
    # X - matrix with signals organized by rows
    #win_size - size of the standard window where the search will bemade
    # len_signal = np.shape(X)[0]
    # nbr_signals = np.shape(X)[1]

    #main words assigned to features
    keywords_keys = ['noise', 'up', 'down', 'flat', 'symmetric', 'assymetric', 'complex', 'high', 'low', 'peak',
                     'valley', 'stepup','stepdown', 'plateauup', 'plateaudown', 'top', 'bottom', 'simple', 'quick', 'vval', 'uval',
                     'middle', 'clean']

    # containers where word feature vectors will be stored and assigned to words
    KEY_VECTOR1 = multi_d_scale_vector_generation(X, int(win_size), keywords_keys)
    #window 2 = win_size / 4
    KEY_VECTOR2 = multi_d_scale_vector_generation(X, int(win_size / 4), keywords_keys)
    #window 3 = win_size / 8
    KEY_VECTOR3 = multi_d_scale_vector_generation(X, int(win_size / 8), keywords_keys)

    return KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3


def word_feature_extraction(X, win_size):
    if(np.ndim(X)==1):
        KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3 = uni_dimensional_extraction(X, win_size)
        return KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3
    elif(np.ndim(X)>1):
        KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3 = multi_dimensional_extraction(X, win_size)
        return KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3
    else:
        print("dimensions are not correct")

