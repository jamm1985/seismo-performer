# Seismo-Performer

In this repository we release implementation of the model, the model configuration, the pretrained best-fitted weights and the code examples from the original paper _The Seismo-Performer: A Novel Machine Learning Approach for General and Efficient Seismic Phase Recognition from Local Earthquakes in Real Time_.


<!-- vim-markdown-toc GFM -->

* [Installation](#installation)
* [Available models](#available-models)
* [How to predict on archives](#how-to-predict-on-archives)
    * [Input file](#input-file)
    * [Output file](#output-file)
    * [Usage Examples](#usage-examples)
    * [Options](#options)
    * [Custom models](#custom-models)
      * [Custom model example](#custom-model-example)
* [Test datasets](#test-datasets)
* [Models training and validation](#models-training-and-validation)
* [Citation](#citation)

<!-- vim-markdown-toc -->

![Model overview](https://www.mdpi.com/sensors/sensors-21-06290/article_deploy/html/images/sensors-21-06290-g001.png)

# Installation

Clone:
```
git clone https://github.com/jamm1985/seismo-performer.git
cd seismo-performer
```

The code is well tested in Python 3.8 and tensorflow 2.5.0 and 2.4.1. 

In existing environments, install the following package versions:
```
pip install einops==0.3.0 obspy==1.2.2 kapre==0.3.5 tensorflow==2.5.0
```

You can also use virtual environment:

```
pip install virtualenv
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

# Available models

**Seismo-Performer** - high performance fast-attention spectrogram-based model with 57к parameters.

**Spec-CNN** - spectrogram-based model variant with CNN instead of Performer (176k parameters).

**GPD-fixed** - [Original GPD](https://github.com/interseismic/generalized-phase-detection) model redeployed in tensorflow 2.5 with 1,742k parameters.


# How to predict on archives

`python archive_scan.py [OPTIONS] <input_file>`

Seismo-Performer model prediction example:
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
NYSH P 0.9997605681 01.04.2021 12:35:55.66
NYSH S 0.9996142387 01.04.2021 12:36:04.46
NYSH P 0.9997449517 01.04.2021 19:19:18.86
```

### Usage Examples

Scan archives using regular high performance fast-attention model, with detection threshold `0.9997` for P and `0.9995` S waves:

```
python archive_scan.py test/nysh_archives.txt --threshold "p: 0.9997, s: 0.9995" --time --print-precision 10
```

To speed up processing on GPU please increase batch size (200000+) and trace size (6000+). You can also turn off the preprocessing filter (2 Hz), as it was for the original training data. 

CNN model variant:

```
python archive_scan.py test/nysh_archives.txt --cnn --threshold "p: 0.9999, s: 0.9995" --time --print-precision 10
```

[Original GPD](https://github.com/interseismic/generalized-phase-detection) model redeployed in tensorflow 2.5:

```
python archive_scan.py --gpd test/nysh_archives.txt --threshold "p: 0.9997, s: 0.9995" --time --print-precision 10
```

Display help message:
<br>
```
python archive_scan.py -h
```

### Options
`-h`, `--help` - display help message
<br>`--weights`, `-w` FILENAME - path to model weights file
<br>`--cnn` - use CNN model variant
<br>`--out`, `-o` FILENAME - output file, default: *predictions.txt*
<br>`--threshold` VALUE - positive prediction threshold, default: *0.95*;
<br> threshold can be also customized per label, usage example: `--threshold "p:0.95, s:0.99"`;
threshold string format: *"[label:threshold],..."*
<br>`--trace-size` VALUE Length of loaded and processed seismic data stream, default: 600 seconds
<br>`--batch-size` VALUE - model batch size, default: 150 slices 
(generally each slice is: 4 seconds by 3 channels)
<br>`--shift` VALUE - sliding window shift in samples, default: *10* milliseconds. Increase in
value will produce faster results, but with potential loss of prediction accuracy. Values above
*200* are not recommended.
<br>`--no-filter` - Do not filter input waveforms
<br>`--no-detrend` - Do not detrend input waveforms
<br>`--print-precision` PRECISION - Floating point precision for predictions pseudo-probability output
<br>`--time` - Print model prediction performance time (in stdout)
<br>`--cpu` - Enforce only CPU resources usage
<br>`--trace-normalization` - Normalize input data per trace (see `--trace-size`). By default, per window
normalization used. Using per trace normalization will reduce memory usage and yield a very small increase in
performance at cost of potentially lower detection accuracy (original models are trained for per window normalization)


### Custom models
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

[Sakhalin (HDF5, 93MB)](https://drive.google.com/file/d/1dH2JF9TQmyB6GpIB_dY1jiWAI5uqp6ED/view?usp=sharing): Total samples 9827: P 3045, S 3737, Noise 3045.

[Dagestan (HDF5, 373 MB)](https://drive.google.com/file/d/156w3I9QVnhkCo0u7wjh-c6xekE9f6B3G/view?usp=sharing): Total samples 28111: P 9547, S 9017, Noise 9547.

Each dataset contains contains seismograms evenly split between P-waves, S-waves, and noise classes. 

Sample dims:
- `X shape: 400x3` (4-second, 3-channel, 100 sps wave data)
- `Y shape: 1` (sparse class labels: `P=0`, `S=1`, `Noise=2`)

# Models training and validation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jamm1985/seismo-performer/blob/main/seismo_performer.ipynb)

Please, see examples of training, testing, and validation at [seismo-performer.ipynb](https://github.com/jamm1985/seismo-performer/blob/main/seismo_performer.ipynb).

# Citation

Stepnov, A.; Chernykh, V.; Konovalov, A. The Seismo-Performer: A Novel Machine Learning Approach for General and Efficient Seismic Phase Recognition from Local Earthquakes in Real Time. Sensors 2021, 21, 6290. https://doi.org/10.3390/s21186290

BibTeX:

```
@Article{s21186290,
AUTHOR = {Stepnov, Andrey and Chernykh, Vladimir and Konovalov, Alexey},
TITLE = {The Seismo-Performer: A Novel Machine Learning Approach for General and Efficient Seismic Phase Recognition from Local Earthquakes in Real Time},
JOURNAL = {Sensors},
VOLUME = {21},
YEAR = {2021},
NUMBER = {18},
ARTICLE-NUMBER = {6290},
URL = {https://www.mdpi.com/1424-8220/21/18/6290},
ISSN = {1424-8220},
ABSTRACT = {When recording seismic ground motion in multiple sites using independent recording stations one needs to recognize the presence of the same parts of seismic waves arriving at these stations. This problem is known in seismology as seismic phase picking. It is challenging to automate the accurate picking of seismic phases to the level of human capabilities. By solving this problem, it would be possible to automate routine processing in real time on any local network. A new machine learning approach was developed to classify seismic phases from local earthquakes. The resulting model is based on spectrograms and utilizes the transformer architecture with a self-attention mechanism and without any convolution blocks. The model is general for various local networks and has only 57 k learning parameters. To assess the generalization property, two new datasets were developed, containing local earthquake data collected from two different regions using a wide variety of seismic instruments. The data were not involved in the training process for any model to estimate the generalization property. The new model exhibits the best classification and computation performance results on its pre-trained weights compared with baseline models from related work. The model code is available online and is ready for day-to-day real-time processing on conventional seismic equipment without graphics processing units.},
DOI = {10.3390/s21186290}
}
```
