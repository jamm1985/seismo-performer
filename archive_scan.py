import argparse
import numpy as np
from numpy.lib.npyio import load
from obspy import read
import obspy.core as oc
from scipy.signal import find_peaks

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import seismo_transformer as st
from tensorflow import keras

import sys


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


def cut_traces(*_traces):
    """
    Cut traces to same timeframe (same start time and end time). Returns list of new traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)
    """
    _start_time = max([x.stats.starttime for x in _traces])
    _end_time = max([x.stats.endtime for x in _traces])

    return_traces = [x.slice(_start_time, _end_time) for x in _traces]

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

    for _i in range(win_count):

        _start_pos = _i * n_shift
        _end_pos = _start_pos + n_features

        windows[_i][:] = data[_start_pos : _end_pos]

    return windows.copy()


def normalize_windows_global(windows):
    """
    Normalizes sliding windows array. IMPORTANT: windows should have separate memory, striped windows would break.
    :param windows:
    :return:
    """
    # Shape (windows_number, n_features, channels_number)
    n_win = windows.shape[0]
    ch_num = windows.shape[2]

    for _i in range(n_win):

        win_max = np.max(np.abs(windows[_i, :, :]))
        windows[_i, :, :] = windows[_i, :, :] / win_max


def normalize_windows_per_trace(windows):
    """
    Normalizes sliding windows array. IMPORTANT: windows should have separate memory, striped windows would break.
    :param windows:
    :return:
    """
    # Shape (windows_number, n_features, channels_number)
    n_win = windows.shape[0]
    ch_num = windows.shape[2]

    for _i in range(n_win):

        for _j in range(ch_num):

            win_max = np.max(np.abs(windows[_i, :, _j]))
            windows[_i, :, _j] = windows[_i, :, _j] / win_max


def scan_traces(*_traces, model = None, n_features = 400, shift = 10, batch_size = 100):
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
    for x in _traces:
        if type(x) != oc.trace.Trace:
            raise TypeError('traces should be a list or containing obspy.core.trace.Trace objects')

    # Cut all traces to a same timeframe
    _traces = cut_traces(*_traces)

    # Normalize
    # TODO: Change normalization to normalization per element
    # normalize_traces(*traces, global_normalize = global_normalize)

    # Get sliding window arrays
    l_windows = []
    for x in _traces:
        l_windows.append(sliding_window(x.data, n_features = n_features, n_shift = shift))

    w_length = min([x.shape[0] for x in l_windows])

    # Prepare data
    windows = np.zeros((w_length, n_features, len(l_windows)))

    for _i in range(len(l_windows)):
        windows[:, :, _i] = l_windows[_i][:w_length]

    # Global max normalization:
    normalize_windows_global(windows)

    # Per-channel normalization:
    # normalize_windows_per_trace(windows)

    # Predict
    _scores = model.predict(windows, verbose = False, batch_size = batch_size)

    # Plot
    # if args and args.plot_positives:
    #     plot_threshold_scores(scores, windows, params['threshold'], file_name, params['plot_labels'])

    # Save scores
    # if args and args.save_positives:
    #     save_threshold_scores(scores, windows, params['threshold'],
    #                           params['positives_h5_path'], params['save_h5_labels'])

    return _scores


def restore_scores(_scores, shape, shift):
    """
    Restores scores to original size using linear interpolation.

    Arguments:
    scores -- original 'compressed' scores
    shape  -- shape of the restored scores
    shift  -- sliding windows shift
    """
    new_scores = np.zeros(shape)
    for i in range(1, _scores.shape[0]):

        for j in range(_scores.shape[1]):

            start_i = (i - 1) * shift
            end_i = i * shift
            if end_i >= shape[0]:
                end_i = shape[0] - 1

            new_scores[start_i : end_i, j] = np.linspace(_scores[i - 1, j], _scores[i, j], shift + 1)[:end_i - start_i]

    return new_scores


def get_positives(_scores, peak_idx, other_idxs, peak_dist = 10000, avg_window_half_size = 100, min_threshold = 0.8):
    """
    Returns positive prediction list in format: [[sample, pseudo-probability], ...]
    """
    _positives = []

    x = _scores[:, peak_idx]

    peaks = find_peaks(x, distance = peak_dist, height=[min_threshold, 1.])

    for _i in range(len(peaks[0])):

        start_id = peaks[0][_i] - avg_window_half_size
        if start_id < 0:
            start_id = 0

        end_id = start_id + avg_window_half_size*2
        if end_id > len(x):
            end_id = len(x) - 1
            start_id = end_id - avg_window_half_size*2

        # Get mean values
        peak_mean = x[start_id : end_id].mean()

        means = []
        for idx in other_idxs:
            means.append(_scores[:, idx][start_id : end_id].mean())

        is_max = True
        for m in means:

            if m > peak_mean:
                is_max = False

        if is_max:
            _positives.append([peaks[0][_i], peaks[1]['peak_heights'][_i]])

    return _positives


def truncate(f, n):
    """
    Floors float to n-digits after comma.
    """
    import math
    return math.floor(f * 10 ** n) / 10 ** n


def print_results(_detected_peaks, filename):
    """
    Prints out peaks in the file.
    """
    with open(filename, 'a') as f:

        for record in _detected_peaks:

            line = ''
            # Print wave type
            line += f'{record["type"]} '

            # Print pseudo-probability
            line += f'{truncate(record["pseudo-probability"], 2):1.2f} '

            # Print time
            dt_str = record["datetime"].strftime("%d.%m.%Y %H:%M:%S")
            line += f'{dt_str}\n'

            # Write
            f.write(line)


def parse_archive_csv(path):
    """
    Parses archives names file. Returns list of filename lists: [[archive1, archive2, archive3], ...]
    :param path:
    :return:
    """
    with open(path) as f:
        lines = f.readlines()

    _archives = []
    for line in lines:
        _archives.append([x for x in line.split()])

    return _archives


def load_transformer(weights_path):
    """
    Loads standard ST model.
    :param weights_path: Path to weights file.
    :return:
    """
    _model = st.seismo_transformer(maxlen = 400,
                                   patch_size = 25,
                                   num_channels = 3,
                                   d_model = 48,
                                   num_heads = 8,
                                   ff_dim_factor = 4,
                                   layers_depth = 8,
                                   num_classes = 3,
                                   drop_out_rate = 0.1)

    _model.load_weights(weights_path)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])

    return _model


def load_favor(weights_path):
    """
    Loads fast-attention ST model variant.
    :param weights_path:
    :return:
    """
    _model = st.seismo_performer_with_spec(maxlen=400,
                                            nfft=128,
                                            patch_size_1=35,
                                            patch_size_2=13,
                                            num_channels=3,
                                            num_patches=5,
                                            d_model=48,
                                            num_heads=4,
                                            ff_dim_factor=4,
                                            layers_depth=2,
                                            num_classes=3,
                                            drop_out_rate=0.1)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    _model.load_weights(weights_path)

    return _model

def load_cnn(weights_path):
    """
    Loads CNN model on top of spectrogram.
    :param weights_path:
    :return:
    """
    _model = st.model_cnn_spec(400,128)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    _model.load_weights(weights_path)

    return _model


if __name__ == '__main__':

    # TODO: PARSE ARGUMENTS BEFORE LOADING TENSORFLOW

    # Command line arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = 'Path to .csv file with archive names')
    parser.add_argument('--weights', '-w', help = 'Path to model weights', default = None)
    parser.add_argument('--favor', help = 'Use Fast-Attention Seismo-Transformer variant', action = 'store_true')
    parser.add_argument('--cnn', help = 'Use simple CNN model on top of spectrogram', action = 'store_true')
    parser.add_argument('--model', help = 'Custom model loader import, default: None', default = None)
    parser.add_argument('--loader_argv', help = 'Custom model loader arguments, default: None', default = None)
    parser.add_argument('--out', '-o', help = 'Path to output file with predictions', default = 'predictions.txt')
    parser.add_argument('--threshold', help = 'Positive prediction threshold, default: 0.95', default = 0.95)
    parser.add_argument('--verbose', '-v', help = 'Provide this flag for verbosity', action = 'store_true')
    parser.add_argument('--batch_size', '-b', help = 'Batch size, default: 500000 samples', default = 500000)

    args = parser.parse_args()  # parse arguments

    # Validate arguments
    if not args.model and not args.weights:

        parser.print_help()
        print()
        sys.stderr.write('ERROR: No --weights specified, either specify --weights argument or use'
                         ' custom model loader with --model flag!')
        sys.exit(2)

    # Set default variables
    # TODO: make them customisable through command line arguments
    model_labels = {'P': 0, 'S': 1, 'N': 2}
    positive_labels = {'P': 0, 'S': 1}

    frequency = 100.

    args.threshold = float(args.threshold)
    args.batch_size = int(args.batch_size)

    archives = parse_archive_csv(args.input)  # parse archive names

    # Load model
    if args.model:

        # TODO: Check if loader_argv is set and check (if possible) loader_call if it receives arguments
        #       Print warning then if loader_argv is not set and print help message about custom models

        import importlib

        model_loader = importlib.import_module(args.model)  # import loader module
        loader_call = getattr(model_loader, 'load_model')  # import loader function

        # Parse loader arguments
        loader_argv = args.loader_argv

        # TODO: Improve parsing to support quotes and whitespaces inside said quotes
        #       Also parse whitespaces between argument and key
        argv_split = loader_argv.strip().split()
        argv_dict = {}

        for pair in argv_split:

            spl = pair.split('=')
            if len(spl) == 2:
                argv_dict[spl[0]] = spl[1]

        model = loader_call(**argv_dict)

    elif args.cnn:
        model = load_cnn(args.weights)
    elif not args.favor:
        model = load_transformer(args.weights)
    else:
        model = load_favor(args.weights)

    # Main loop
    for n_archive, l_archives in enumerate(archives):

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

        n_traces = len(streams[0])

        # Progress bar preparations
        total_batch_count = 0
        for i in range(n_traces):

            traces = [st[i] for st in streams]

            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % args.batch_size
            batch_count = l_trace // args.batch_size + 1 \
                if last_batch \
                else l_trace // args.batch_size

            total_batch_count += batch_count

        # Predict
        current_batch_global = 0
        for i in range(n_traces):

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

                # Progress bar
                progress_bar(current_batch_global / total_batch_count, 40, add_space_around = False,
                             prefix = f'Group {n_archive + 1} out of {len(archives)} [',
                             postfix = f'] - Batch: {batches[0].stats.starttime} - {batches[0].stats.endtime}')
                current_batch_global += 1

                scores = scan_traces(*batches, model = model, batch_size = args.batch_size)  # predict

                if scores is None:
                    continue

                # TODO: window step 10 should be in params, including the one used in predict.scan_traces
                restored_scores = restore_scores(scores, (len(batches[0]), len(model_labels)), 10)

                # Get indexes of predicted events
                predicted_labels = {}
                for label in positive_labels:

                    other_labels = []
                    for k in model_labels:
                        if k != label:
                            other_labels.append(model_labels[k])

                    positives = get_positives(restored_scores,
                                              positive_labels[label],
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
                        tmp_prediction_dates.append([starttime + (prediction[0] / frequency), prediction[1]])

                    predicted_timestamps[label] = tmp_prediction_dates

                # Prepare output data
                for typ in predicted_timestamps:
                    for pred in predicted_timestamps[typ]:

                        prediction = {'type': typ,
                                      'datetime': pred[0],
                                      'pseudo-probability': pred[1]}

                        detected_peaks.append(prediction)

                print_results(detected_peaks, args.out)

            print('')
