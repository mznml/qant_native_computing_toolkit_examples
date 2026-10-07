# Diffusion Model
This project demonstrates an image-conditional denoising diffusion model that turns photos of fruits into pixel art, executed on the Q.ANT Native Computing Toolkit.

We here use the Q.ANT Native Computing Toolkit to implement PyTorch layers that allow training on CPU / GPU and straightforward evaluation on the NPU.
A basic understanding of PyTorch and diffusion models is recommended.

## Overview
Notebook: diffusion_on_npu.ipynb \
Contains the complete workflow for training a denoising network and sampling from it on the NPU.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Run the notebook:
Open diffusion_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_diffusion.py executes the notebook and checks that training converges.

```bash
pytest test_diffusion.py
```

# Description
A denoising diffusion model (DDPM) with 100 steps learns to turn random noise into the pixel art of a given photo.
The denoising network is a small MLP built from `QLinear` layers (see `qant_layers.py`): it is trained with standard PyTorch
on the CPU, and in evaluation mode its matrix multiplications could run on the NPU (if available) during sampling.
Its input is the noisy pixel art, a step embedding and the photo (downscaled to 32x32); it predicts the clean pixel art.

## Dataset
The training data is built from a few classes of the Fruits-360 dataset (https://github.com/fruits-360/fruits-360-100x100)
by Mihai Oltean, licensed under CC BY-SA 4.0. The selected classes (~3.8k images) are downloaded at runtime
(training and test split) with a sparse git checkout into `dataset/fruits-360`, so `git` is required. The dataset is not redistributed with this example.

Each photo is converted into 16x16 pixel art in three steps (see `pixel_art_data.py`): sharpen, quantize with a shared
16-color k-means palette, and grid encode (each cell of the 16x16 grid takes its most frequent palette color).
The photo is the condition of the diffusion model and its pixel art the target. Photos of the test split are only used
to generate pixel art after training, to show that the model works on photos it has never seen.
