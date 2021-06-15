import argparse


if __name__ == '__main__':

    # Options parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', help = 'Use CNN version of the Seismo-Transformer',
                        action = 'store_true')
    parser.add_argument('--favor', help = 'Use Fast-Attention version of the Seismo-Transformer',
                        action = 'store_true')
    parser.add_argument('--model', help = 'Custom model loader module import path')
    parser.add_argument('--data', '-d', help = 'Dataset file path')
    parser.add_argument('--out', '-o', help = 'Output file path')
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

    print('SCORES:')
    print(scores)

    # Save predictions info to .csv
