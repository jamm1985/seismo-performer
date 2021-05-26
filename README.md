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
separated by a whitespace and daily archive groups separated by newlines.

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
<br>`--favor` - use fast attention model variant
<br>`--out`, `-o` FILENAME - output file, default: *predictions.txt*
<br>`--threshold` VALUE - positive prediction threshold, default: *0.95*
<br>`--batch_size` VALUE - batch size, default: *500 000* samples

### Model selection
To select fast-attention variant of the Seismo Transformer use --favor flag. 
Regular Seismo Transformer will be used otherwise.

### Examples
Scan archives using fast-attention model, with detection threshold 0.98:
<br>`python archive_scan.py test/nysh_archives.txt WEIGHTS/sakh_favor_2014_2019.h5 --favor --threshold 0.98`

Scan archives using regular model, with detection threshold 0.95:
<br>`python archive_scan.py test/nysh_archives.txt WEIGHTS/sakh_favor_2014_2019.h5 --favor --threshold 0.98`

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
