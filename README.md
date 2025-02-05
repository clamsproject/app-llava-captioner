# LLaVA Captioner

## Description

The LLaVA Captioner is a CLAMS app designed to generate textual descriptions for video frames or images using the LLaVA v1.6 Mistral-7B model. 

For more information about LLaVA see: [LLaVA Project Page](https://llava-vl.github.io/)

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

Below is a list of additional information specific to this app.

### System requirements

The preferred platform is Debian 10.13 or higher. GPU is not required but performance will be significantly better with it. The main system packages needed are FFmpeg (https://ffmpeg.org/), OpenCV4 (https://opencv.org/), and Python 3.8 or higher.

The easiest way to get these is to get the Docker clams-python-opencv4 base image. For more details take a peek at the following container specifications for the CLAMS base, FFMpeg and OpenCV containers. Python packages needed are: clams-python, ffmpeg-python, opencv-python-rolling, transformers, torch, and Pillow. Some of these are installed on the Docker clams-python-opencv4 base image and some are listed in requirements.txt in this repository.

### Configurable Runtime Parameters

The app supports the following parameters:

- `frameInterval` (integer, default: 30): The interval at which to extract frames from the video if there are no timeframe annotations.
- `defaultPrompt` (string): Default prompt to use for timeframe types not specified in the promptMap.
- `promptMap` (map): Mapping of labels of input timeframe annotations to specific prompts. Format: "IN_LABEL:PROMPT".
- `config` (string, default: "config/default.yaml"): Path to the configuration file.

### Configuration Files

The app supports YAML configuration files to specify detailed behavior. Several example configurations are provided:


Each configuration file can specify:
- `default_prompt`: The prompt template to use with LLaVA
- `custom_prompts`: Label-specific prompts for different types of content
- `context_config`: Specifies how to process the input (timeframe, timepoint, fixed_window, or image)

For specific use cases, see the example configuration files in the `config/` directory:
- `fixed_window.yaml`: Regular interval processing
- `shot_captioning.yaml`: Shot-based video captioning
- `slate_dates_images.yaml`: Date extraction from slates
- `slates_all_fields.yaml`: Detailed metadata extraction from slates
- `swt_transcription.yaml`: Text transcription with custom prompts for different frame types

