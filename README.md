# Seismo-Transformer

# Introduction

In this repository we release implementation of the model, the model configuration, the pretrained best-fitted weights and the code examples from the original paper _The Seismo-Transformer: generally based machine learning approach for recognizing seismic phases from local earthquakes_. 

# Installation

# Available models

# How to fine-tune

# Test datasets

[Sakhalin (HDF5, 127 MB)](https://drive.google.com/file/d/1dH2JF9TQmyB6GpIB_dY1jiWAI5uqp6ED/view?usp=sharing). Total samples 13689: P 4137, S 4776, Noise 4776.

[Dagestan (HDF5, 200 MB)](https://drive.google.com/file/d/156w3I9QVnhkCo0u7wjh-c6xekE9f6B3G/view?usp=sharing). Total samples 21572: P 8068, S 6752, Noise 6752.

Each dataset contains contains seismograms evenly split between P-waves, S-waves, and noise classes. 

Sample dims:
- `X shape: 400x3` (4-second, 3-channel, 100 sps wave data)
- `Y shape: 1` (class label: P=0, S=1, Noise=2)

# Citation

The manuscript have been submitted to a journal.
