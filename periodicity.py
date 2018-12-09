import datetime
import math
from collections import Counter

import numpy as np
import pandas as pd


def format_seconds(num):
    """
    string formatting
    note that the days in a month is kinda fuzzy
    :type num: int | float
    """
    num = abs(num)
    if num == 0:
        return u'0 seconds'
    elif num == 1:
        return u'1 second'

    if num < 1:
        # display 2 significant figures worth of decimals
        return (u'%%0.%df seconds' % (1 - int(math.floor(math.log10(abs(num)))))) % num

    unit = 0
    denominators = [60.0, 60.0, 24.0, 7.0, 365.25 / 84.0, 12.0]
    while unit < 6 and num > denominators[unit] * 0.9:
        num /= denominators[unit]
        unit += 1
    unit = [u'seconds', u'minutes', u'hours', u'days', u'weeks', u'months', u'years'][unit]
    return (u'%.2f %s' if num % 1 else u'%d %s') % (num, unit[:-1] if num == 1 else unit)


def flatten_timestamps(events, magnitudes=None):
    if magnitudes is None:
        magnitudes = [1] * len(events)

    # convert datetime to unix seconds
    epoch = datetime.datetime(1970, 1, 1)
    unix_events = [int((dt - epoch).total_seconds()) for dt in events if not pd.isna(dt)]

    # anchors
    unix_first = min(unix_events)
    unix_last = max(unix_events)

    # make sure the output length isn't a small multiple of a big prime
    seq_len = 1 + unix_last - unix_first
    seq_len_bit_reduce = max(seq_len.bit_length() - 8, 2)
    seq_len = ((seq_len >> seq_len_bit_reduce) + 1) << seq_len_bit_reduce

    # flatten
    time_series = [0] * seq_len
    for unix_event, magnitude in zip(unix_events, magnitudes):
        time_series[unix_event - unix_first] += magnitude

    return time_series


def fourier(time_series):
    # fourier frequencies and complex amplitudes
    frequencies = np.fft.fftfreq(len(time_series))
    amplitudes = np.fft.fft(time_series)

    # we don't want the negative frequencies
    frequencies = frequencies[:len(frequencies) / 2]
    amplitudes = amplitudes[:len(amplitudes) / 2]

    # we don't want the zero frequency == infinite period
    frequencies = frequencies[1:]
    amplitudes = amplitudes[1:]

    # convert to periods and non-complex magnitudes
    periods = (1 / frequencies)[::-1]
    magnitudes = np.absolute(amplitudes)[::-1]

    # bin by second and take max
    last_period = 0.0
    last_magnitude = 0
    out_periods = []
    out_magnitudes = []
    for p, m in zip(periods, magnitudes):
        if round(p) == last_period:
            last_magnitude = max(last_magnitude, m)
        else:
            out_periods.append(last_period)
            out_magnitudes.append(last_magnitude)
            last_period = round(p)
            last_magnitude = m
    out_periods.append(int(last_period))
    out_magnitudes.append(last_magnitude)

    return out_periods, out_magnitudes


def threshold_time_series(time_series, verbose=False):
    threshold = max(x for x, y in Counter(time_series).most_common() if y != 1)

    count_removed = 0
    for i in range(len(time_series)):
        if time_series[i] > threshold:
            count_removed += 1
            time_series[i] = threshold

    if verbose:
        print(u'set threshold = {}; trimmed {} data points'.format(threshold, count_removed))

    return time_series


def analyze(events, magnitudes=None, threshold=1000, min_items=30, verbose=True):
    """

    :param events: list of timestamps (can be unsorted and non-unique)
    :param magnitudes: corresponding list of event magnitude/weight for each timestamp, if applicable
    :param threshold: automatically apply a threshold for outlier events
    :param verbose: something to watch while jupyter is being useless
    :param min_items: if there are fewer than this many unique timestamps, returns NaN
    :return: df of periods and weights
    """
    # convert to events per second (or sum of magnitudes per second)
    if verbose:
        print(u'flattening...')
    time_series = flatten_timestamps(events, magnitudes)

    num_items = sum(x > 0 for x in time_series)
    if verbose:
        print(u'have %d unique timestamps' % num_items)

    # skip if not useful
    if min_items and num_items < min_items:
        if verbose:
            print(u'insufficient unique timestamps, meaningless to fft (less than %d)' % min_items)
        return np.nan

    # very basic automatic thresholding
    if threshold and num_items > threshold:
        if verbose:
            print(u'thresholding...')
        time_series = threshold_time_series(time_series, verbose=verbose)
    elif verbose and threshold:
        print(u'not thresholding, only have %d unique timestamps (less than 1000)' % num_items)

    # run the fft and store results in a df
    if verbose:
        print(u'fft...')
    periods, magnitudes = fourier(time_series)

    if verbose:
        print(u'making dataframe...')
    df_fft = pd.DataFrame.from_records(zip(magnitudes, periods), columns=[u'magnitude', u'period_seconds'])
    df_fft[u'period_days'] = df_fft[u'period_seconds'].apply(lambda x: x / 3600.0 / 24.0)
    df_fft[u'period_timedelta'] = df_fft[u'period_seconds'].apply(lambda x: pd.Timedelta(seconds=x))
    df_fft[u'period_formatted'] = df_fft[u'period_seconds'].apply(format_seconds)
    df_fft[u'weight'] = df_fft[u'magnitude'].apply(np.log)

    return df_fft.sort_values(by=u'magnitude', ascending=False)


def get_beacon_ratio(ts):
    """
    Calculate ratio of records that were beaconing to all records.
    Detect beaconing by checking for consistent inter-arrival times.

    Assumptions:
     - beaconing is at least 50-80% of the overall activity
     - beacons don't span multiple rows over multiple seconds
    """
    ts = ts.sort_values().drop_duplicates()
    timedelta = (ts - ts.shift(1))
    ts_beacon = ts[timedelta.between(timedelta.median() - timedelta.std(),
                                     timedelta.median() + timedelta.std())]
    ratio = len(ts_beacon) / float(len(ts))
    return pd.DataFrame([{u'ratio':       ratio,
                          u'timedelta_s': timedelta.median().total_seconds(),
                          u'timedelta_f': format_seconds(timedelta.median().total_seconds())
                          }])


def get_beacon_ratio_records(df, timestamp_col_name=u'timestamp'):
    """
    Return records that were involved in beaconing
    Detect beaconing by checking for consistent inter-arrival times.

    Assumptions:
     - beaconing is at least 50-80% of the overall activity
     - beacons don't span multiple rows over multiple seconds
    """
    ts = df[timestamp_col_name]
    timedelta = (ts - ts.shift(1))
    ts_beacon = ts[timedelta.between(timedelta.median() - timedelta.std(),
                                     timedelta.median() + timedelta.std())]
    return df.loc[ts_beacon.index]


def get_topk_periods(timestamps, k=20, verbose=True, min_items=30):
    """
    :param timestamps: list of timestamps
    :param k: how many rows to return for each group
    :param verbose: print(stuff to monitor progress from jupyter)
    :return: df
    """
    out = analyze(timestamps.dropna(), verbose=verbose, min_items=min_items)
    if type(out) is float and pd.isna(out):
        return pd.DataFrame()
    return out.sort_values(by=u'magnitude', ascending=False).head(k)
