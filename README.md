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

The app supports YAML configuration files to specify detailed behavior. Several example configurations are provided in the `config/` directory:

Each configuration file can specify:
- `default_prompt`: The prompt template to use with LLaVA
- `custom_prompts`: Label-specific prompts for different types of content
- `context_config`: Specifies how to process the input (timeframe, timepoint, fixed_window, or image)

#### Example Configuration Files

- `fixed_window.yaml`: Regular interval processing
- `shot_captioning.yaml`: Shot-based video captioning
- `slate_dates_images.yaml`: Date extraction from slates
- `slates_all_fields.yaml`: Detailed metadata extraction from slates
- `swt_transcription.yaml`: Text transcription with custom prompts for different frame types

### Usage Examples

#### Basic Usage with CLI

```bash
# Process a video with default settings
python cli.py input.mmif output.mmif

# Use a specific configuration file
python cli.py --config config/swt_transcription.yaml input.mmif output.mmif

# Use custom prompts
python cli.py --defaultPrompt "Describe this frame in detail" --promptMap "slate:Extract all text from this slate" input.mmif output.mmif
```

#### Using with Docker

```bash
# Build the container
docker build -t llava-captioner .

# Run the container
docker run -v /path/to/data:/data llava-captioner python cli.py /data/input.mmif /data/output.mmif
```

#### Web Service

```bash
# Start the web service
python app.py --port 5000

# In production mode
python app.py --production
```

### Input/Output Specifications

#### Input
- **VideoDocument**: Video files to extract frames from
- **ImageDocument**: Individual images to caption
- **TimeFrame**: Annotations specifying time segments to process

#### Output
- **TextDocument**: Generated captions for each processed frame
- **Alignment**: Links between TimePoint annotations and generated text documents


