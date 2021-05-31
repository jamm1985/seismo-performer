# Seismo-Transformer

# Introduction

In this repository we release implementation of the model, the model configuration, the pretrained best-fitted weights and the code examples from the original paper _The Seismo-Transformer: generally based machine learning approach for recognizing seismic phases from local earthquakes_. 

# Installation

# Available models

# How to fine-tune

# How to predict on archives

`python archive_scan.py [OPTIONS] <input_file> <model_weights_path>`

### Input file
`input_file` contains archive filenames (recommended channel order: `N, E, Z`) 
separated by a whitespace. Every new archive group should be separated by a newline.

`input_file` example:

```
test/archives/NYSH.IM.00.EHN.2021.091 test/archives/NYSH.IM.00.EHE.2021.091 test/archives/NYSH.IM.00.EHZ.2021.091
test/archives/NYSH.IM.00.EHN.2021.092 test/archives/NYSH.IM.00.EHE.2021.092 test/archives/NYSH.IM.00.EHZ.2021.092
```

Note that files are passed to the model in the order they are specified in `input_file`. 
Advised channel order: `N, E, Z`.

### Output file
Output file consists of positives predictions divided by a line break.

Prediction format:
<br>`<phase_hint> <pseudo-probability> <date> <time>`
where date is in format: `<day>.<month>.<year>`.

Example:
```
P 0.99 01.04.2021 22:11:00
S 0.99 01.04.2021 21:13:16
S 0.99 01.04.2021 21:38:54
```

### Options
`-h` - display help message
<br>`--weights`, `-w` FILENAME - path to model weights file
<br>`--favor` - use fast attention model variant
<br>`--out`, `-o` FILENAME - output file, default: *predictions.txt*
<br>`--threshold` VALUE - positive prediction threshold, default: *0.95*
<br>`--batch_size` VALUE - batch size, default: *500 000* samples
<br>`--no-filter` - Do not filter input waveforms
<br>`--no-detrend` - Do not detrend input waveforms

### Model selection and custom models

#### Seismo-Transformer

To select fast-attention variant of the Seismo-Transformer use --favor flag. 
Regular Seismo-Transformer will be used otherwise.

#### Custom models
It is possible to predict with custom models, in order to do so, follow these steps:

*1. Create model loader*

Model loader is basically a function with name `load_model` inside a python module.
For example `test/keras_loader.py` provides a custom loader for tensorflow.keras models, 
saved in *.json* format:

```aidl
import tensorflow as tf
from tensorflow.keras.models import model_from_json


def load_model(model_path, weights_path):

    print('Keras loader call!')
    print('model_path = ', model_path)
    print('weights_path = ', weights_path)

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects = {"tf": tf})

    model.load_weights(weights_path)

    return model
```

*2. Use --model option with `archive_scan.py` call*

Using *--model* option followed by loader module import path will let the script know, 
that using a custom model loader is required.
Function `load_model` then will be called.
`load_model` arguments can be provided using *--loader_argv* option.
*--loader_argv* should be followed by a string of `key=value` pairs separated by a whitespace.

#### Custom model example
`python .\archive_scan.py --model test.keras_loader --loader_argv "model_path=path/to/model weights_path=path/to/weights" .\test\nysh_archives.txt`

### Examples
Scan archives using fast-attention model, with detection threshold 0.98:
<br>`python archive_scan.py test/nysh_archives.txt -w WEIGHTS/sakh_favor_2014_2019.h5 --favor --threshold 0.98`

Scan archives using regular model, with detection threshold 0.95:
<br>`python archive_scan.py test/nysh_archives.txt -w WEIGHTS/sakh_favor_2014_2019.h5 --favor --threshold 0.98`

Using keras .json model:
<br>`python .\archive_scan.py --model test.keras_loader --loader_argv "model_path=path/to/model weights_path=path/to/weights" .\test\nysh_archives.txt`

Display help message:
<br>`python archive_scan.py -h`


# Test datasets

[Sakhalin (HDF5, 127 MB)](https://drive.google.com/file/d/1dH2JF9TQmyB6GpIB_dY1jiWAI5uqp6ED/view?usp=sharing). Total samples 13689: P 4137, S 4776, Noise 4776.

[Dagestan (HDF5, 200 MB)](https://drive.google.com/file/d/156w3I9QVnhkCo0u7wjh-c6xekE9f6B3G/view?usp=sharing). Total samples 21572: P 8068, S 6752, Noise 6752.

Each dataset contains contains seismograms evenly split between P-waves, S-waves, and noise classes. 

Sample dims:
- `X shape: 400x3` (4-second, 3-channel, 100 sps wave data)
- `Y shape: 1` (class label: P=0, S=1, Noise=2)

# Citation

The manuscript have been submitted to a journal.
