import argparse
import numpy as np
from obspy import read
import sys
from obspy.core.utcdatetime import UTCDateTime

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    # Default weights for models
    default_weights = {'favor': 'WEIGHTS/w_model_performer_with_spec.hd5',
                       'hpa': 'WEIGHTS/w_model_performer_with_spec_hight_accuracy.hd5',
                       'cnn': 'WEIGHTS/weights_model_cnn_spec.hd5',
                       'gpd': 'WEIGHTS/w_gpd_scsn_2000_2017.h5'}

    # Command line arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = 'Path to .csv file with archive names')
    parser.add_argument('--weights', '-w', help = 'Path to model weights', default = None)
    parser.add_argument('--hpa', help = 'Use Fast-Attention with high accuracy', action = 'store_true')
    parser.add_argument('--cnn', help = 'Use simple CNN model on top of spectrogram', action = 'store_true')
    parser.add_argument('--gpd', help = 'Use GPD model', action = 'store_true')
    parser.add_argument('--model', help = 'Custom model loader import, default: None', default = None)
    parser.add_argument('--loader_argv', help = 'Custom model loader arguments, default: None', default = None)
    parser.add_argument('--out', '-o', help = 'Path to output file with predictions', default = 'predictions.txt')
    parser.add_argument('--threshold', help = 'Positive prediction threshold, default: 0.95', default = 0.95)
    parser.add_argument('--batch-size', help = 'Model batch size, default: 150 slices '
                                               '(each slice is: 4 seconds by 3 channels)',
                        default = 150)
    parser.add_argument('--trace-size', '-b', help = 'Length of loaded and processed seismic data stream, '
                                                     'default: 600 seconds', default = 600)
    parser.add_argument('--shift', help = 'Sliding windows shift, default: 10 samples (10 ms)', default = 10)
    parser.add_argument('--no-filter', help = 'Do not filter input waveforms', action = 'store_true')
    parser.add_argument('--no-detrend', help = 'Do not detrend input waveforms', action = 'store_true')
    parser.add_argument('--plot-positives', help = 'Plot positives waveforms', action = 'store_true')
    parser.add_argument('--plot-positives-original', help = 'Plot positives original waveforms, before '
                                                            'pre-processing',
                        action = 'store_true')
    parser.add_argument('--print-scores', help = 'Prints model prediction scores and according wave forms data'
                                                 ' in .npy files',
                        action = 'store_true')
    parser.add_argument('--print-precision', help = 'Floating point precision for results pseudo-probability output',
                        default = 4)
    parser.add_argument('--time', help = 'Print out performance time in stdout', action = 'store_true')
    parser.add_argument('--cpu', help = 'Disable GPU usage', action = 'store_true')
    parser.add_argument('--start', help = 'Earliest time stamp allowed for input waveforms,'
                                          ' format examples: "2021-04-01" or "2021-04-01T12:35:40"', default = None)
    parser.add_argument('--end', help = 'Latest time stamp allowed for input waveforms'
                                        ' format examples: "2021-04-01" or "2021-04-01T12:35:40"', default = None)
    parser.add_argument('--trace-normalization', help = 'Normalize input data per trace, otherwise - per full trace.'
                                                        ' Increases performance and reduces memory demand if set (at'
                                                        ' a cost of potential accuracy loss).',
                        action = 'store_true')

    args = parser.parse_args()  # parse arguments

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set label variables
    # TODO: make them customisable through command line arguments
    model_labels = {'p': 0, 's': 1, 'n': 2}
    positive_labels = {'p': 0, 's': 1}
    # TODO: Change threshold_s and threshold_s so they would be dynamic parameter --threshold
    #   e.g. '--threshold "p 0.92, s 0.98"'

    # Parse and validate thresholds
    threshold_labels = {}
    global_threshold = False
    if type(args.threshold) is str:

        split_thresholds = args.threshold.split(',')

        if len(split_thresholds) == 1:
            args.threshold = float(args.threshold)
            global_threshold = True
        else:
            for split in split_thresholds:

                label_threshold = split.split(':')
                if len(label_threshold) != 2:

                    parser.print_help()
                    sys.stderr.write('ERROR: Wrong --threshold format. Hint:'
                                     ' --threshold "p: 0.95, s: 0.9901"')
                    sys.exit(2)

                threshold_labels[label_threshold[0].strip()] = float(label_threshold[1])
    else:
        args.threshold = float(args.threshold)
        global_threshold = True

    if global_threshold:
        for label in positive_labels:
            threshold_labels[label] = args.threshold
    else:
        positive_labels_error = False
        if len(positive_labels) != len(threshold_labels):
            positive_labels_error = True

        for label in positive_labels:
            if label not in threshold_labels:
                positive_labels_error = True

        if positive_labels_error:
            parser.print_help()
            sys.stderr.write('ERROR: --threshold values do not match positive_labels.'
                             f' positive_labels contents: {[k for k in positive_labels.keys()]}')
            sys.exit(2)

    # Set start and end date
    def parse_date_param(args, p_name):
        """
        Parse parameter from dictionary to UTCDateTime type.
        """
        if not getattr(args, p_name):
            return None

        try:
            return UTCDateTime(getattr(args, p_name))
        except TypeError as e:
            print(f'Failed to parse "{p_name}" parameter (value: {getattr(args, p_name)}).'
                  f' Use {__file__} -h for date format information.')
            sys.exit(1)
        except Exception as e:
            print(f'Failed to parse "{p_name}" parameter (value: {getattr(args, p_name)}).'
                  f' Use {__file__} -h for date format information.')
            raise

    args.end = parse_date_param(args, 'end')
    args.start = parse_date_param(args, 'start')

    # Set values
    frequency = 100.
    n_features = 400
    half_duration = (n_features * 0.5) / frequency

    args.batch_size = int(args.batch_size)
    args.trace_size = int(float(args.trace_size) * frequency)
    args.shift = int(args.shift)
    args.print_precision = int(args.print_precision)

    import utils.scan_tools as stools

    archives = stools.parse_archive_csv(args.input)  # parse archive names

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
    # TODO: Print loaded model info. Also add flag --inspect to print model summary.
    else:

        if args.cnn:
            import utils.seismo_load as seismo_load
            if not args.weights: args.weights = default_weights['cnn']
            model = seismo_load.load_cnn(args.weights)
        elif args.hpa:
            import utils.seismo_load as seismo_load
            if not args.weights: args.weights = default_weights['hpa']
            model = seismo_load.load_performer_hpa(args.weights)
        elif args.gpd:
            from utils.gpd_loader import load_model as load_gpd
            if not args.weights: args.weights = default_weights['gpd']
            model = load_gpd(args.weights)
        else:
            import utils.seismo_load as seismo_load
            if not args.weights: args.weights = default_weights['favor']
            model = seismo_load.load_performer(args.weights)

    # Main loop
    total_performance_time = 0.
    for n_archive, l_archives in enumerate(archives):

        # Write archives info
        with open(args.out, 'a') as f:
            line = ''
            for path in l_archives:
                line += f'{path} '
            line += '\n'
            f.write(line)

        # Read data
        streams = []
        for path in l_archives:
            streams.append(read(path))

        # If --plot-positives-original, save original streams
        original_streams = None
        if args.plot_positives_original:
            original_streams = []
            for path in l_archives:
                original_streams.append(read(path))

        # Pre-process data
        for st in streams:
            stools.pre_process_stream(st, args.no_filter, args.no_detrend)

        # Cut archives to the same length
        streams = stools.trim_streams(streams, args.start, args.end)
        if original_streams:
            original_streams = stools.trim_streams(original_streams, args.start, args.end)

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
            last_batch = l_trace % args.trace_size
            batch_count = l_trace // args.trace_size + 1 \
                if last_batch \
                else l_trace // args.trace_size

            total_batch_count += batch_count

        # Predict
        current_batch_global = 0
        for i in range(n_traces):

            traces = stools.get_traces(streams, i)
            original_traces = None
            if original_streams:
                original_traces = stools.get_traces(original_streams, i)
                if traces[0].data.shape[0] != original_traces[0].data.shape[0]:
                    raise AttributeError('WARNING: Traces and original_traces have different sizes, '
                                         'check if preprocessing changes stream length!')

            # Determine batch count
            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % args.trace_size
            batch_count = l_trace // args.trace_size + 1 \
                if last_batch \
                else l_trace // args.trace_size

            freq = traces[0].stats.sampling_rate
            station = traces[0].stats.station

            for b in range(batch_count):

                detected_peaks = []

                b_size = args.trace_size
                if b == batch_count - 1 and last_batch:
                    b_size = last_batch

                start_pos = b * args.trace_size
                end_pos = start_pos + b_size
                t_start = traces[0].stats.starttime

                batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq) for trace in traces]
                original_batches = None
                if original_traces:
                    original_batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq)
                                        for trace in original_traces]

                # Progress bar

                if args.time:
                    stools.progress_bar(current_batch_global / total_batch_count, 40, add_space_around = False,
                                        prefix = f'Group {n_archive + 1} out of {len(archives)} [',
                                        postfix = f'] - Batch: {batches[0].stats.starttime} '
                                                  f'- {batches[0].stats.endtime} '
                                                  f'Time: {total_performance_time:.6} seconds')
                else:
                    stools.progress_bar(current_batch_global / total_batch_count, 40, add_space_around = False,
                                        prefix = f'Group {n_archive + 1} out of {len(archives)} [',
                                        postfix = f'] - Batch: {batches[0].stats.starttime}'
                                                  f' - {batches[0].stats.endtime}')
                current_batch_global += 1

                scores, performance_time = stools.scan_traces(*batches,
                                                              model = model,
                                                              args = args,
                                                              original_data = original_batches)  # predict
                total_performance_time += performance_time

                if scores is None:
                    continue

                # TODO: window step 10 should be in params, including the one used in predict.scan_traces
                restored_scores = stools.restore_scores(scores, (len(batches[0]), len(model_labels)), args.shift)

                # Get indexes of predicted events
                predicted_labels = {}
                for label in positive_labels:

                    other_labels = []
                    for k in model_labels:
                        if k != label:
                            other_labels.append(model_labels[k])

                    positives = stools.get_positives(restored_scores,
                                                     positive_labels[label],
                                                     other_labels,
                                                     threshold = threshold_labels[label])

                    predicted_labels[label] = positives

                # Convert indexes to datetime
                predicted_timestamps = {}
                for label in predicted_labels:

                    tmp_prediction_dates = []
                    for prediction in predicted_labels[label]:

                        starttime = batches[0].stats.starttime

                        # Get prediction UTCDateTime and model pseudo-probability
                        # TODO: why params['frequency'] here but freq = traces[0].stats.frequency before?
                        tmp_prediction_dates.append([starttime + (prediction[0] / frequency) + half_duration,
                                                     prediction[1]])

                    predicted_timestamps[label] = tmp_prediction_dates

                # Prepare output data
                for typ in predicted_timestamps:
                    for pred in predicted_timestamps[typ]:

                        prediction = {'type': typ,
                                      'datetime': pred[0],
                                      'pseudo-probability': pred[1]}

                        detected_peaks.append(prediction)

                if args.print_scores:
                    stools.print_scores(batches, restored_scores, predicted_labels, f't{i}_b{b}')

                stools.print_results(detected_peaks, args.out, precision = args.print_precision, station = station)

            print('')

        # Write separator
        with open(args.out, 'a') as f:
            line = '---' * 12 + '\n'
            f.write(line)

    if args.time:
        print(f'Total model prediction time: {total_performance_time:.6} seconds')
