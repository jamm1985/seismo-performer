import argparse
import numpy as np
from obspy import read
import obspy.core as oc
from scipy.signal import find_peaks

def pre_process_stream(stream):
    """
    Does preprocessing on the stream (changes it's frequency), does linear detrend and
    highpass filtering with frequency of 2 Hz.

    Arguments:
    stream      -- obspy.core.stream object to pre process
    frequency   -- required frequency
    """
    stream.detrend(type="linear")
    stream.filter(type="highpass", freq = 2)

    frequency = 100.
    required_dt = 1. / frequency
    dt = stream[0].stats.delta

    if dt != required_dt:
        stream.interpolate(frequency)


def progress_bar(progress, characters_count = 20,
                 erase_line = True,
                 empty_bar = '.', filled_bar = '=', filled_edge = '>',
                 prefix = '', postfix = '',
                 add_space_around = True):
    """
    Prints progress bar.
    :param progress: percentage (0..1) of progress, or int number of characters filled in progress bar.
    :param characters_count: length of the bar in characters.
    :param erase_line: preform return carriage.
    :param empty_bar: empty bar character.
    :param filled_bar: progress character.
    :param filled_edge: progress character on the borderline between progressed and empty,
                        set to None to disable.
    :param prefix: progress bar prefix.
    :param postfix: progress bar postfix.
    :param add_space_around: add space after prefix and before postfix.
    :return:
    """

    space_characters = ' \t\n'
    if add_space_around:
        if len(prefix) > 0 and prefix[-1] not in space_characters:
            prefix += ' '

        if len(postfix) > 0 and postfix[0] not in space_characters:
            postfix = ' ' + postfix

    if erase_line:
        print('\r', end = '')

    progress_num = int(characters_count * progress)
    if filled_edge is None:
        print(prefix + filled_bar * progress_num + empty_bar * (characters_count - progress_num) + postfix, end = '')
    else:
        bar_str = prefix + filled_bar * progress_num
        bar_str += filled_edge * min(characters_count - progress_num, 1)
        bar_str += empty_bar * (characters_count - progress_num - 1)
        bar_str += postfix

        print(bar_str, end = '')


def cut_traces(*traces):
    """
    Cut traces to same timeframe (same start time and end time). Returns list of new traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)
    """
    start_time = max([x.stats.starttime for x in traces])
    end_time = max([x.stats.endtime for x in traces])

    return_traces = [x.slice(start_time, end_time) for x in traces]

    return return_traces


def sliding_window(data, n_features, n_shift):
    """
    Return NumPy array of sliding windows. Which is basically a view into a copy of original data array.

    Arguments:
    data       -- numpy array to make a sliding windows on
    n_features -- length in samples of the individual window
    n_shift    -- shift between windows starting points
    """
    # Get sliding windows shape
    win_count = np.floor(data.shape[0]/n_shift - n_features/n_shift + 1).astype(int)
    shape = (win_count, n_features)

    windows = np.zeros(shape)

    for i in range(win_count):

        start_pos = i * n_shift
        end_pos = start_pos + n_features

        windows[i][:] = data[start_pos : end_pos]

    return windows.copy()


def normalize_windows_per_trace(windows):
    """
    Normalizes sliding windows array. IMPORTANT: windows should have separate memory, striped windows would break.
    :param windows:
    :return:
    """
    # Shape (windows_number, n_features, channels_number)
    n_win = windows.shape[0]
    ch_num = windows.shape[2]

    for i in range(n_win):

        for j in range(ch_num):

            win_max = np.max(np.abs(windows[i, :, j]))
            windows[i, :, j] = windows[i, :, j] / win_max


def scan_traces(*traces, model = None, n_features = 400, shift = 10, batch_size = 100, args = None, code = None):
    """
    Get predictions on the group of traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)

    Keyword arguments
    model            -- NN model
    n_features       -- number of input features in a single channel
    shift            -- amount of samples between windows
    global_normalize -- normalize globaly all traces if True or locally if False
    batch_size       -- model.fit batch size
    """
    # Check input types
    for x in traces:
        if type(x) != oc.trace.Trace:
            raise TypeError('traces should be a list or containing obspy.core.trace.Trace objects')

    # Cut all traces to a same timeframe
    traces = cut_traces(*traces)

    # Normalize
    # TODO: Change normalization to normalization per element
    # normalize_traces(*traces, global_normalize = global_normalize)

    # Get sliding window arrays
    l_windows = []
    for x in traces:
        l_windows.append(sliding_window(x.data, n_features = n_features, n_shift = shift))

    w_length = min([x.shape[0] for x in l_windows])

    # Prepare data
    windows = np.zeros((w_length, n_features, len(l_windows)))

    for i in range(len(l_windows)):
        windows[:, :, i] = l_windows[i][:w_length]

    normalize_windows_per_trace(windows)

    # Predict
    scores = model.predict(windows, verbose = False, batch_size = batch_size)

    # Plot
    # if args and args.plot_positives:
    # plot_threshold_scores(scores, windows, params['threshold'], file_name, params['plot_labels'])

    # Save scores
    # if args and args.save_positives:
        # save_threshold_scores(scores, windows, params['threshold'], params['positives_h5_path'], params['save_h5_labels'])

    return scores


def restore_scores(scores, shape, shift):
    """
    Restores scores to original size using linear interpolation.

    Arguments:
    scores -- original 'compressed' scores
    shape  -- shape of the restored scores
    shift  -- sliding windows shift
    """
    new_scores = np.zeros(shape)
    for i in range(1, scores.shape[0]):

        for j in range(scores.shape[1]):

            start_i = (i - 1) * shift
            end_i = i * shift
            if end_i >= shape[0]:
                end_i = shape[0] - 1

            new_scores[start_i : end_i, j] = np.linspace(scores[i - 1, j], scores[i, j], shift + 1)[:end_i - start_i]

    return new_scores


def get_positives(scores, peak_indx, other_indxs, peak_dist = 10000, avg_window_half_size = 100, min_threshold = 0.8):
    """
    Returns positive prediction list in format: [[sample, pseudo-probability], ...]
    """
    positives = []

    x = scores[:, peak_indx]
    peaks = find_peaks(x, distance = peak_dist, height=[min_threshold, 1.])

    for i in range(len(peaks[0])):

        start_id = peaks[0][i] - avg_window_half_size
        if start_id < 0:
            start_id = 0

        end_id = start_id + avg_window_half_size*2
        if end_id > len(x):
            end_id = len(x) - 1
            start_id = end_id - avg_window_half_size*2

        # Get mean values
        peak_mean = x[start_id : end_id].mean()

        means = []
        for indx in other_indxs:

            means.append(scores[:, indx][start_id : end_id].mean())

        is_max = True
        for m in means:

            if m > peak_mean:
                is_max = False

        if is_max:
            positives.append([peaks[0][i], peaks[1]['peak_heights'][i]])

    return positives


def print_results(detected_peaks, filename):
    """
    Prints out peaks in the file.
    """
    with open(filename, 'a') as f:

        for record in detected_peaks:

            line = ''
            # Print wave type
            line += f'{record["type"]} '

            # Print pseudo-probability
            line += f'{truncate(record["pseudo-probability"], 2):1.2f} '

            # Print station
            line += f'{record["station"]} '

            # Print location
            line += f'{record["location_code"]} '

            # Print net code
            line += f'{record["network_code"]}   '

            # Print time
            dt_str = record["datetime"].strftime("%d.%m.%Y %H:%M:%S")
            line += f'{dt_str}   '

            # Print channels
            line += f'{[ch for ch in record["channels"]]}\n'

            # Write
            f.write(line)


if __name__ == '__main__':

    args = object

    # TODO: initialize archive lists [[N_path, E_path, Z_path], ...] into archives variable
    archives = []

    # Main loop
    for l_archives in archives:

        # Read data
        streams = []
        for path in l_archives:
            streams.append(read(path))

        # Pre-process data
        for st in streams:
            pre_process_stream(st)

        # Cut archives to the same length
        max_start_time = None
        min_end_time = None

        for stream in streams:

            current_start = min([x.stats.starttime for x in stream])
            current_end = max([x.stats.endtime for x in stream])

            if not max_start_time:
                max_start_time = current_start
            if not min_end_time:
                min_end_time = current_end

            if current_start > max_start_time:
                max_start_time = current_start
            if current_end < min_end_time:
                min_end_time = current_end

        cut_streams = []
        for st in streams:
            cut_streams.append(st.slice(max_start_time, min_end_time))

        streams = cut_streams

        # Check if stream traces number is equal
        lengths = [len(st) for st in streams]
        if len(np.unique(np.array(lengths))) != 1:
            continue

        # Predict
        n_traces = len(streams[0])
        for i in range(n_traces):

            # TODO: Improve progress bar to render by batch inside predict(...)
            progress_bar(i / n_traces, 40, add_space_around = False)

            traces = [st[i] for st in streams]  # get traces

            # Trim traces to the same length
            start_time = max([trace.stats.starttime for trace in traces])
            end_time = min([trace.stats.endtime for trace in traces])

            for j in range(len(traces)):
                traces[j] = traces[j].slice(start_time, end_time)

            # Determine batch count
            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % args.batch_size
            batch_count = l_trace // args.batch_size + 1 \
                if last_batch \
                else l_trace // args.batch_size

            freq = traces[0].stats.sampling_rate

            for b in range(batch_count):

                detected_peaks = []

                b_size = args.batch_size
                if b == batch_count - 1 and last_batch:
                    b_size = last_batch

                start_pos = b * args.batch_size
                end_pos = start_pos + b_size
                t_start = traces[0].stats.starttime

                batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq) for trace in traces]

                scores = scan_traces(*batches,
                                     model = model, params = args)  # predict

                if scores is None:
                    continue

                # TODO: window step 10 should be in params, including the one used in predict.scan_traces
                restored_scores = restore_scores(scores, (len(batches[0]), len(args.model_labels)), 10)

                # Get indexes of predicted events
                predicted_labels = {}
                for label in args.positive_labels:

                    other_labels = []
                    for k in args.model_labels:
                        if k != label:
                            other_labels.append(args.model_labels[k])

                    positives = get_positives(restored_scores,
                                              args.positive_labels[label],
                                                      other_labels,
                                                      min_threshold = args.threshold)

                    predicted_labels[label] = positives

                # Convert indexes to datetime
                predicted_timestamps = {}
                for label in predicted_labels:

                    tmp_prediction_dates = []
                    for prediction in predicted_labels[label]:
                        starttime = batches[0].stats.starttime

                        # Get prediction UTCDateTime and model pseudo-probability
                        # TODO: why params['frequency'] here but freq = traces[0].stats.frequency before?
                        tmp_prediction_dates.append([starttime + (prediction[0] / params['frequency']), prediction[1]])

                    predicted_timestamps[label] = tmp_prediction_dates

                # Prepare output data
                for typ in predicted_timestamps:
                    for pred in predicted_timestamps[typ]:

                        prediction = {'type': typ,
                                      'datetime': pred[0],
                                      'pseudo-probability': pred[1],
                                      'channels': streams_channels,
                                      'station': archive_data['meta']['station'],
                                      'location_code': archive_data['meta']['location_code'],
                                      'network_code': archive_data['meta']['network_code']}

                        detected_peaks.append(prediction)

                print_results(detected_peaks, args.output_file)