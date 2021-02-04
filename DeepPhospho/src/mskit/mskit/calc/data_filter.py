import numpy as np


def remove_outliers(data):
    data = np.array(data)
    p75 = np.percentile(data, 75)
    p25 = np.percentile(data, 25)
    iqr = p75 - p25
    uif = p75 + iqr * 1.5
    dif = p25 - iqr * 1.5
    data = data[(data >= dif) & (data <= uif)]
    return data
