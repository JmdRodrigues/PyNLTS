import numpy as np
import numpy.fft as fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import matplotlib.pyplot as plt
import rolling

def slidingDotProduct(query, ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.
    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """
    m = len(query)
    n = len(ts)

    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]

    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))

    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

def nd_periodogram(ts, m):
    Y = sliding_window_view(ts, m)
    # Calculate the discrete Fourier transform of the time series
    dft = np.fft.fftn(Y)
    dft = dft[:, :m//2]

    power_spectrum = np.abs(dft)**2
    frequencies = np.linspace(0, 0.5, m//2)

    return frequencies, power_spectrum

def md_spectrogram(ts, m):
    f, t, Sxx = signal.spectrogram(ts, nperseg=ts//m, noverlap=(ts//m)-10)
    ax1 = plt.subplot(211)
    ax1.plot(ts)
    ax2 = plt.subplot(212)
    ax2.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.show()

def mass(query, ts):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.
    Parameters
    ----------
    :query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    :ts: Time series to compare against query.
    """

    #query_normalized = zNormalize(np.copy(query))
    m = len(query)
    q_mean = np.mean(query)
    q_std = np.std(query)
    sum_, mean, std = mov_sum_mean_std(ts, m)
    dot = slidingDotProduct(query, ts)

    #res = np.sqrt(2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std)))
    res = 2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std))

    return res

def movsum(ts,m):
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    segSum = s[m:] - s[:-m]

    return segSum

def mov_max_min(ts, m):
    moving_max = np.array(list(rolling.Max(ts, m)))
    moving_min = np.array(list(rolling.Min(ts, m)))


    return moving_max, moving_min, moving_max-moving_min

def movmean(ts, m):
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]

    return segSum/m

def mov_sum_mean_std(ts,m):
    """
    Calculate the standard deviation within a moving window.
    Parameters
    ----------
    ts: signal.
    m: moving window size.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    return segSum, segSum/m, np.sqrt(segSumSq / m - (segSum/m) ** 2)

def norm_1_sig(sig):
    return (sig-np.min(sig))/(np.max(sig)-np.min(sig))

def z_norm_sig(sig):
    return (sig - np.mean(sig)) / np.std(sig)

def np_slope(x, y):
    p = np.polyfit(x, y, 1)[0]
    return p

def moving_slope(y, win_size):
    X = np.linspace(1, win_size, win_size)
    Y = sliding_window_view(y, win_size)
    return np_slope(X, np.transpose(Y))

def quick_moving_feature_extraction(sig, win_size):
    half_win_size = int(win_size / 2)

    mov_sum, mov_mean, mov_std = mov_sum_mean_std(sig, win_size)

    mov_max, mov_min, mov_dif = mov_max_min(sig, win_size)

    moving_high = norm_1_sig(mov_dif)
    moving_low = 1 - moving_high

    moving_noise = norm_1_sig(mov_std)
    moving_clean = 1-moving_noise

    moving_complexity = np.insert(norm_1_sig(np.sqrt(movsum(np.diff(sig)**2, win_size))), 0,0)
    moving_simple = 1-moving_complexity

    moving_top = norm_1_sig(mov_mean)

    moving_not_middle = np.zeros(len(sig))
    moving_not_middle[:-(win_size-1)] = abs(sig[:-(win_size-1)] - mov_mean)
    moving_not_middle = norm_1_sig(movmean(moving_not_middle, win_size))
    moving_middle = 1 - moving_not_middle

    moving_bottom = 1 - moving_top

    moving_sl = moving_slope(sig, win_size)
    moving_up = norm_1_sig(np.where(moving_sl>0, moving_sl, 0))
    moving_down = norm_1_sig(abs(np.where(moving_sl<0, moving_sl, 0)))
    # moving_flat = 1 - norm_1_sig(sum(abs(sig - np.mean(sig)), 1))
    moving_flat = 1 - norm_1_sig(abs(moving_down)+moving_up)

    moving_speed = np.insert(norm_1_sig(movsum(abs(np.diff(sig)), win_size)), 0,0)
    moving_complexity = np.insert(norm_1_sig(np.sqrt(movsum(np.diff(sig) ** 2, win_size))), 0, 0)

    return moving_noise, moving_complexity, moving_simple, moving_clean, moving_high, moving_low, moving_bottom, moving_top, moving_middle, moving_up, moving_down, moving_speed, moving_flat

def slow_moving_feature_extraction(sig, win_size):
    symmetric = moving_symmetry(sig, win_size)

    return symmetric

def gauss_peak_moving(sig, c, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)
    query_peak = a * np.exp(((x - b)**2) / (-2 * (c**2)))
    mass_peak = np.log10(mass(query_peak, sig))
    mass_peak = 1 - norm_1_sig(mass_peak)

    return mass_peak

def gauss_val_moving(sig, c, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)
    query_valley_ = -a*np.exp(((x-b)**2)/(-2*(c**2)))
    mass_val = np.log10(mass(query_valley_, sig))
    mass_val = 1 - norm_1_sig(mass_val)

    return mass_val

def v_gauss_val_moving(sig, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)

    valley_ = -a * np.exp(((x - b)**2) / (-2 * (0.1**2)))

    mass_v_valley = np.log10(mass(valley_, sig))
    # mass_v_valley(isinf(mass_v_valley)) = max(mass_v_valley(~isinf(mass_v_valley)));
    # mass_v_valley(isnan(mass_v_valley)) = max(mass_v_valley(~isnan(mass_v_valley)));
    mass_v_valley = 1 - norm_1_sig(mass_v_valley)

    return mass_v_valley

def u_gauss_val_moving(sig, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)

    valley_ = -a * np.exp(((x - b)**2) / (-2 * (1**2)))

    mass_u_valley = np.log10(mass(valley_, sig))
    # mass_v_valley(isinf(mass_v_valley)) = max(mass_v_valley(~isinf(mass_v_valley)));
    # mass_v_valley(isnan(mass_v_valley)) = max(mass_v_valley(~isnan(mass_v_valley)));
    mass_u_valley = 1 - norm_1_sig(mass_u_valley)

    return mass_u_valley

def moving_symmetry(sig, win_size):
    sig = (sig - np.mean(sig))/np.std(sig)
    moving_sig = sliding_window_view(sig, win_size)
    fliped_sig = np.fliplr(moving_sig)

    symmetry = 1 - norm_1_sig(np.sqrt(np.sum((moving_sig-fliped_sig)**2, axis=1)))

    return symmetry

def moving_step(sig, win_size):
    half_win_size = int(win_size / 2)

    pos_step_query = np.zeros(win_size)
    neg_step_query = np.zeros(win_size)

    pos_step_query[win_size // 2:] = 1
    neg_step_query[win_size // 2:] = -1

    pos_step = mass(pos_step_query, sig)
    pos_step = 1 - norm_1_sig(pos_step)
    neg_step = mass(neg_step_query, sig)
    neg_step = 1 - norm_1_sig(neg_step)

    return pos_step, neg_step

def moving_plateau(sig, win_size):
    pos_plat_query = np.zeros(win_size)
    neg_plat_query = np.zeros(win_size)

    pos_plat_query[win_size //4: 3 * win_size // 4] = 1
    neg_plat_query[win_size //4: 3 * win_size // 4] = -1

    pos_plateau = mass(pos_plat_query, sig)
    pos_plateau = 1 - norm_1_sig(pos_plateau)
    neg_plateau = mass(neg_plat_query, sig)
    neg_plateau = 1 - norm_1_sig(neg_plateau)

    return pos_plateau, neg_plateau

# def get_features(ts, m):
#     moving_noise, moving_complexity, moving_simple, moving_clean, moving_high, moving_low, moving_bottom, moving_top, moving_middle = quick_moving_feature_extraction(ts, m)


    # x_mat = [np.ones(win_size, 1), np.transpose(np.linspace(1, win_size, win_size))]

    # slopes_ = x_mat\mirror_sig_mat
    # slopes_ = slopes_(2,:)
    #
    # moving_up = norm_1_sig(max(slopes_, 0));
    # moving_down = norm_1_sig(abs(min(slopes_, 0)));
    #
    # moving_flat = norm_1_sig(sum(abs(mirror_sig_mat - mean(mirror_sig_mat, 1)), 1));
    # moving_flat = 1 - moving_flat;
