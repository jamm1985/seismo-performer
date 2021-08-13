import argparse
import h5py as h5
import numpy as np
import pandas as pd

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    # Options parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', help = 'Path to Seismo-Performer model weights file')
    parser.add_argument('--cnn', help = 'Use CNN version of the Seismo-Performer',
                        action = 'store_true')
    parser.add_argument('--favor', help = 'Use Fast-Attention version of the Seismo-Performer',
                        action = 'store_true')
    parser.add_argument('--model', help = 'Custom model loader module import path')
    parser.add_argument('--data', '-d', help = 'Dataset file path')
    parser.add_argument('--out', '-o', help = 'Output file path', default = 'prc_out.csv')
    parser.add_argument('--loader-argv', help = 'Output file path')

    args = parser.parse_args()

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

        import utils.seismo_load as seismo_load

        if args.cnn:
            model = seismo_load.load_cnn(args.weights)
        elif args.favor:
            model = seismo_load.load_favor(args.weights)
        else:
            model = seismo_load.load_transformer(args.weights)

    # Load data with h5_generator
    from h5_generator import train_test_split as h5_tts

    _, X_test = h5_tts(args.data, batch_size = 100, shuffle = False, train_size = 0.)

    # Predict
    scores = model.predict(X_test)

    # Read labels
    with h5.File(args.data, 'r') as f:
        Y = np.array(f['Y'], dtype = 'int')

    # Get predictions
    Y_pred = np.argmax(scores, axis = 1)
    Y_scores = np.max(scores, axis = 1)

    # Save predictions info to .csv
    data = {
        'Y_true': Y,
        'Y_pred': Y_pred,
        'Y_score': Y_scores
    }

    df = pd.DataFrame(data)
    df.to_csv(args.out, index = False)
