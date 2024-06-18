from datetime import datetime
import os
import numpy as np

def convert_array2str(ar, sep=" "):
    r = ["%.7f" % ar[i] for i in range(len(ar))]
    return sep.join(r)
def get_fft(signal,norm="ortho"):
    sampling_rate = 240  # in Hz

    # Step 2: Compute the FFT of the signal
    fft_result = np.fft.rfft(signal, norm="ortho")

    # Step 3: Compute the corresponding frequencies
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)

    # Step 4: Select the desired frequencies and extract their corresponding FFT values
    # Let's say we are interested in the frequencies: 50 Hz, 120 Hz, and 200 Hz
    desired_freqs = np.arange(0, 60, 0.25)
    desired_indices = [np.argmin(np.abs(frequencies - f)) for f in desired_freqs]
    fft_values_at_desired_freqs = fft_result[desired_indices]
    return abs(fft_values_at_desired_freqs)


def get_insert_dict_index(d, k):
    try:
        v = d[k]
    except:
        v = len(d)
        d[k] = v
    return v

def convert_time(time_string, offset=946659600000):
    FORMAT1 = '%Y.%m.%d.  %H:%M:%S.%f'
    FORMAT11 = '%Y.%m.%d.  %H:%M:%S'
    if time_string.__contains__("/"):
        FORMAT1 =  '%m/%d/%Y  %H:%M:%S.%f'
        FORMAT11 = '%m/%d/%Y  %H:%M:%S.%f'

    try:
        if time_string[-5:].__contains__("."):
            dt_obj = datetime.strptime(time_string,
                                   FORMAT1)
        else:
            dt_obj = datetime.strptime(time_string,
                                   FORMAT11)

        millisec = int(dt_obj.timestamp() * 1000) - offset
    except:
        millisec = -1
    return millisec

def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)