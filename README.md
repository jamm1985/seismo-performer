# Seismo-Transformer

# Introduction

In this repository we release implementation of the model, the model configuration, the pretrained best-fitted weights and the code examples from the original paper _The Seismo-Transformer: generally based machine learning approach for recognizing seismic phases from local earthquakes_. 

# Installation

# Available models

# How to fine-tune

# How to predict on archives

`python archive_scan.py [OPTIONS] <input_file>`

Performer model prediction example:
<br>
```
python archive_scan.py test/nysh_archives.txt
```

Predictions are saved in text file which default name is `predictions.txt`

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
<br>
```
<station> <phase_hint> <pseudo-probability> <date> <time>
```

Example:
```
NYSH P 0.9994 01.04.2021 00:04:28.46
NYSH P 0.9994 01.04.2021 00:10:57.06
NYSH S 0.9998 01.04.2021 00:04:52.96
NYSH S 0.9991 01.04.2021 00:08:39.56
NYSH P 0.9997 01.04.2021 00:31:05.36
```

### Usage Examples

Scan archives using regular high performance fast-attention model, with detection threshold `0.9999` for P and `0.9995` S waves:

```
python archive_scan.py test/nysh_archives.txt --threshold "p: 0.9997, s: 0.9995" --batch-size 10000 --time --print-precision 10
```

To speed up processing on GPU please increase batch size (100000+). You can also turn off the preprocessing filter (2 Hz), as it was for the original training data. Note that in the case of Seismo-Performer, less preprocessing does not actually affect prediction performance, so the same prediction thresholds can be used:

```
python archive_scan.py test/nysh_archives.txt --threshold "p: 0.9997, s: 0.9995" --batch-size 100000 --no-detrend --time --print-precision 10
```

CNN model variant:

```
python archive_scan.py test/nysh_archives.txt --cnn --threshold "p: 0.9999, s: 0.9995" --batch-size 10000 --time --print-precision 10
```

To improve time for scanning acrhives please increase batch size.

[GPD](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/108/5A/2894/546740/Generalized-Seismic-Phase-Detection-with-Deep?redirectedFrom=fulltext) model:
```
python archive_scan.py --gpd test/nysh_archives.txt
```

Display help message:
<br>
```
python archive_scan.py -h
```

### Options
`-h`, `--help` - display help message
<br>`--weights`, `-w` FILENAME - path to model weights file
<br>`--hpm` - use fast attention model with high accuracy
<br>`--cnn` - use CNN model variant
<br>`--out`, `-o` FILENAME - output file, default: *predictions.txt*
<br>`--threshold` VALUE - positive prediction threshold, default: *0.95*;
<br> threshold can be also customized per label, usage example: `--threshold "p:0.95, s:0.99"`;
threshold string format: *"[label:threshold],..."*
<br>`--batch-size` VALUE - batch size, default: *10 000* samples
<br>`--no-filter` - Do not filter input waveforms
<br>`--no-detrend` - Do not detrend input waveforms
<br>`--print-precision` PRECISION - Floating point precision for predictions pseudo-probability output
<br>`--time` - Print model prediction performance time (in stdout)
<br>`--cpu` - Enforce only CPU resources usage

### Model selection and custom models

#### Seismo-Performer

To select fast-attention with high accuracy use `--hpm` flag.
<br>Use `--cnn` flag for CNN model variant.
<br>High perfomance Seismo-Performer will be used if no flag provided.

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
```
python .\archive_scan.py --model test.keras_loader --loader_argv "model_path=path/to/model weights_path=path/to/weights" .\test\nysh_archives.txt
```


# Test datasets

[Sakhalin (HDF5, 127 MB)](https://drive.google.com/file/d/1dH2JF9TQmyB6GpIB_dY1jiWAI5uqp6ED/view?usp=sharing). Total samples 13689: P 4137, S 4776, Noise 4776.

[Dagestan (HDF5, 200 MB)](https://drive.google.com/file/d/156w3I9QVnhkCo0u7wjh-c6xekE9f6B3G/view?usp=sharing). Total samples 21572: P 8068, S 6752, Noise 6752.

Each dataset contains contains seismograms evenly split between P-waves, S-waves, and noise classes. 

Sample dims:
- `X shape: 400x3` (4-second, 3-channel, 100 sps wave data)
- `Y shape: 1` (class label: P=0, S=1, Noise=2)

# Citation

The manuscript have been submitted to a journal.
