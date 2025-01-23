# Llama Captioner

## Description

The Llama Captioner is a CLAMS app designed to generate textual descriptions for video frames using the LLaMA 3.2 model. It processes video documents with timeframe annotations and generates captions accordingly.

For more information about LLaMA 3.2 see: [LLaMA 3.2 Blog Post](https://llama-vl.github.io/blog/2024-01-30-llama-3.2/)

For LLaMA license information see: [LLaMA Github Repo](https://github.com/haotian-liu/LLaMA#:~:text=Usage%20and%20License,laws%20and%20regulations.)

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

Below is a list of additional information specific to this app.

- Currently the app uses a hard-coded prompt and does not accept any app-specific parameters. A future version of the app will accept custom prompts via a config file. 

### System requirements

The preferred platform is Debian 10.13 or higher. GPU is not required but performance will be significantly better with it. The main system packages needed are FFmpeg (https://ffmpeg.org/), OpenCV4 (https://opencv.org/), and Python 3.8 or higher.

The easiest way to get these is to get the Docker clams-python-opencv4 base image. For more details take a peek at the following container specifications for the CLAMS base, FFMpeg and OpenCV containers. Python packages needed are: clams-python, ffmpeg-python, opencv-python-rolling, transformers, torch, and Pillow. Some of these are installed on the Docker clams-python-opencv4 base image and some are listed in requirements.txt in this repository.

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py) file in this repository.
