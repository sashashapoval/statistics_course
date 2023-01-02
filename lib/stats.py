import numpy as np
import math

#Empirical probability density function
#returns two arrays: centers of the bins and pdf-values
def pdf_my(array, n_bins = 10, arr_min = [], arr_max = []):
    array = np.array(array)
    if np.size(arr_min) == 0:
        arr_min = min(array)
    if np.size(arr_max) == 0:
        arr_max = max(array)
    if n_bins == 0:
        raise Exception('At least 1 bin required')
    else:
        x = []
        y = []
        gap = (arr_max - arr_min) / n_bins
        for i in range(n_bins):
            x.append(arr_min + (0.5 + i) * gap)
            y.append(0)
        n_els = 0
        for arr in array:
            cur = int(math.floor((arr - arr_min) / gap))
            if cur >= 0 and cur < n_bins:
                y[cur] += 1
                n_els += 1
        if n_els > 0:
            for cur in range(len(y)):
                y[cur] = y[cur] / n_els / gap
    return x, y
