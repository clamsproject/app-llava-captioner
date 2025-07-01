import argparse
import logging
import yaml
from pathlib import Path
import tqdm
import time
from PIL import Image

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
import torch


class LlavaCaptioner(ClamsApp):
    """
    A CLAMS app that uses LLaVA v1.6 Mistral-7B model to generate captions for video frames or images.
    
    This app can process:
    - Individual images (ImageDocument)
    - Video frames based on TimeFrame annotations
    - Video frames at fixed intervals
    - Video frames at regular timepoints
    
    The app supports custom prompts for different types of content and can be configured
    via YAML configuration files.
    """

    # Default values for video processing
    DEFAULT_FPS = 29.97
    DEFAULT_VIDEO_DURATION_MINUTES = 1
    
    DEFAULT_MAX_NEW_TOKENS = 100
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_REPETITION_PENALTY = 1.5
    DEFAULT_LENGTH_PENALTY = 1.0

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        try:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf", 
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            self.logger.warning(f"Failed to load model with flash attention: {e}")
            self.logger.info("Falling back to default attention implementation...")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf", 
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="eager",
            )
        
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        super().__init__()

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def load_config(self, config_file: Path) -> dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is invalid YAML
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, parameters: dict) -> str:
        """
        Get the appropriate prompt for a given label based on parameters.
        
        Args:
            label: The label to get a prompt for
            parameters: Dictionary containing prompt mappings and defaults
            
        Returns:
            The prompt string to use for the given label
        """
        # Check if there's a promptMap parameter with mappings
        if 'promptMap' in parameters and parameters['promptMap']:
            # promptMap is a list of "label:prompt" strings
            for mapping in parameters['promptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        
        # Fall back to defaultPrompt if no mapping found
        if 'defaultPrompt' in parameters:
            return parameters['defaultPrompt']
        
        # Final fallback
        return ""

    def _clean_generated_text(self, generated_text: str) -> str:
        """
        Clean generated text by removing instruction tags.
        
        Args:
            generated_text: Raw generated text from the model
            
        Returns:
            Cleaned text without instruction tags
        """
        start_tag = "[INST]"
        end_tag = "[/INST]"
        start_idx = generated_text.find(start_tag)
        end_idx = generated_text.find(end_tag) + len(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return generated_text[end_idx:].strip()
        else:
            return generated_text.strip()

    def _get_video_properties(self, video_doc: Document) -> tuple[float, int]:
        """
        Extract FPS and total frame count from video document.
        
        Args:
            video_doc: Video document to extract properties from
            
        Returns:
            Tuple of (fps, total_frames)
        """
        try:
            fps = float(video_doc.get_property('fps'))
        except (TypeError, ValueError):
            self.logger.warning(f"Could not extract FPS from video document, using default: {self.DEFAULT_FPS}")
            fps = self.DEFAULT_FPS

        try:
            total_frames = int(video_doc.get_property('frameCount'))
        except (TypeError, ValueError):
            self.logger.warning(f"Could not extract frame count from video document, using default")
            total_frames = int(fps * 60 * self.DEFAULT_VIDEO_DURATION_MINUTES)

        return fps, total_frames

    def _process_batch(self, prompts_batch: list, images_batch: list, annotations_batch: list, new_view: View) -> None:
        """
        Process a batch of prompts and images through the LLaVA model.
        
        Args:
            prompts_batch: List of prompts for the batch
            images_batch: List of PIL images for the batch
            annotations_batch: List of annotation metadata for the batch
            new_view: MMIF view to add new annotations to
        """
        try:
            inputs = self.processor(
                images=images_batch[0], 
                text=prompts_batch[0], 
                padding=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.DEFAULT_MAX_NEW_TOKENS,
                min_length=1,
                repetition_penalty=self.DEFAULT_REPETITION_PENALTY,
                length_penalty=self.DEFAULT_LENGTH_PENALTY,
                temperature=self.DEFAULT_TEMPERATURE,
            )
            
            generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            for generated_text, prompt, annotation in zip(generated_texts, prompts_batch, annotations_batch):
                self.logger.debug(f"Generated text: {generated_text}")
                self.logger.debug(f"Prompt: {prompt}")
                
                clean_text = self._clean_generated_text(generated_text)
                self.logger.debug(f"Clean text: {clean_text}")
                
                text_document = new_view.new_textdocument(clean_text)
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", annotation['source'])
                alignment.add_property("target", text_document.long_id)
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise
        finally:
            # Clean up GPU memory
            try:
                del inputs
                del outputs
                del generated_texts
            except NameError:
                pass
            torch.cuda.empty_cache()

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        """
        Main annotation method that processes the input MMIF and generates captions.
        
        Args:
            mmif: Input MMIF object containing documents and annotations
            **parameters: Runtime parameters for the annotation process
            
        Returns:
            Updated MMIF object with new caption annotations
        """
        self.logger.debug(f"Annotating with parameters: {parameters}")
        
        # Load configuration
        config = self._load_configuration(parameters)
        
        # Create new view for output
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)
        
        # Process based on input context
        input_context = config['context_config']['input_context']
        
        if input_context == "image":
            self._process_image_documents(mmif, new_view, parameters)
        elif input_context == 'timeframe':
            self._process_timeframe_annotations(mmif, new_view, parameters, config)
        elif input_context == 'fixed_window':
            self._process_fixed_window(mmif, new_view, parameters, config)
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        return mmif

    def _load_configuration(self, parameters: dict) -> dict:
        """
        Load and validate configuration from parameters or config file.
        
        Args:
            parameters: Runtime parameters
            
        Returns:
            Configuration dictionary
        """
        config_file = parameters.get('config')
        
        if config_file:
            config_dir = Path(__file__).parent
            config_file_path = config_dir / config_file
            config = self.load_config(config_file_path)
            
            # Populate defaultPrompt from config if it exists
            if 'default_prompt' in config:
                parameters['defaultPrompt'] = config['default_prompt']
            
            # Populate promptMap from custom_prompts if it exists
            if 'custom_prompts' in config:
                prompt_map = []
                for label, prompt in config['custom_prompts'].items():
                    prompt_map.append(f"{label}:{prompt}")
                parameters['promptMap'] = prompt_map
        else:
            config = {}
        
        return config

    def _process_image_documents(self, mmif: Mmif, new_view: View, parameters: dict) -> None:
        """
        Process ImageDocument inputs.
        
        Args:
            mmif: Input MMIF object
            new_view: Output view to add annotations to
            parameters: Runtime parameters
        """
        image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
        batch_size = parameters.get('batch_size')
        
        for i in range(0, len(image_docs), batch_size):
            batch_docs = image_docs[i:i + batch_size]
            prompts = [self.get_prompt('default', parameters)] * len(batch_docs)
            images = [Image.open(doc.location_path()) for doc in batch_docs]
            annotations_batch = [{'source': doc.long_id} for doc in batch_docs]
            
            start_time = time.time()
            self._process_batch(prompts, images, annotations_batch, new_view)
            self.logger.info(f"Processed batch of {len(batch_docs)} images in {time.time() - start_time:.2f} seconds")

    def _process_timeframe_annotations(self, mmif: Mmif, new_view: View, parameters: dict, config: dict) -> None:
        """
        Process TimeFrame annotations to extract and caption representative frames.
        
        Args:
            mmif: Input MMIF object
            new_view: Output view to add annotations to
            parameters: Runtime parameters
            config: Configuration dictionary
        """
        self.logger.debug(f"Processing timeframe annotations")
        
        # Find timeframe annotations from the specified app
        app_uri = config['context_config']['timeframe']['app_uri']
        timeframes = None
        
        for view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame):
            if app_uri in view.metadata.app:
                self.logger.debug(f"Found view with app_uri: {app_uri}")
                timeframes = list(view.get_annotations(AnnotationTypes.TimeFrame))
                break
        
        if not timeframes:
            self.logger.warning(f"No timeframe annotations found from app: {app_uri}")
            return
        
        # Filter timeframes based on configuration
        label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
        ignore_other_labels = config['context_config']['timeframe'].get('ignore_other_labels', False)
        
        if ignore_other_labels:
            timeframes = [tf for tf in timeframes if tf.get_property('label') in label_mapping]
            if not timeframes:
                self.logger.warning("No timeframes found with labels matching the label_mapping")
                return
        
        # Add timeUnit property to timeframes
        for timeframe in timeframes:
            timeframe.add_property('timeUnit', 'milliseconds')

        # Extract representative frames
        all_frame_numbers = [vdh.get_representative_framenum(mmif, timeframe) for timeframe in timeframes]
        self.logger.info(f"Extracted {len(all_frame_numbers)} frame numbers")
        
        video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        if not video_doc:
            raise ValueError("No video document found in MMIF")
        
        # Extract frames as images
        try:
            all_images = vdh.extract_frames_as_images(video_doc, all_frame_numbers, as_PIL=True)
            self.logger.info(f"Successfully extracted {len(all_images)} images")
            
            if len(all_images) != len(all_frame_numbers):
                self.logger.warning(f"Number of extracted images ({len(all_images)}) doesn't match number of frame numbers ({len(all_frame_numbers)})")
                
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise
        
        # Process in batches
        batch_size = parameters.get('batch_size')
        for i in tqdm.tqdm(range(0, len(timeframes), batch_size)):
            batch_timeframes = timeframes[i:i + batch_size]
            batch_images = all_images[i:i + batch_size]
            
            # Prepare batch data
            prompts = []
            annotations_batch = []
            for timeframe in batch_timeframes:
                label = timeframe.get_property('label')
                mapped_label = label_mapping.get(label, 'default')
                prompt = self.get_prompt(mapped_label, parameters)
                prompts.append(prompt)
                
                representative_id = timeframe.get_property('representatives')[0]
                annotations_batch.append({'source': representative_id})
            
            start_time = time.time()
            self._process_batch(prompts, batch_images, annotations_batch, new_view)
            self.logger.info(f"Processed batch of {len(batch_timeframes)} timeframes in {time.time() - start_time:.2f} seconds")

    def _process_fixed_window(self, mmif: Mmif, new_view: View, parameters: dict, config: dict) -> None:
        """
        Process video at fixed time intervals.
        
        Args:
            mmif: Input MMIF object
            new_view: Output view to add annotations to
            parameters: Runtime parameters
            config: Configuration dictionary
        """
        self.logger.debug(f"Processing fixed window")
        
        video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        window_duration = config['context_config']['fixed_window']['window_duration']
        stride = config['context_config']['fixed_window']['stride']
        
        fps, total_frames = self._get_video_properties(video_doc)
        frame_numbers = list(range(0, total_frames, int(fps * stride)))
        
        prompts = []
        images_batch = []
        annotations_batch = []
        batch_size = parameters.get('batch_size')
        
        for frame_number in tqdm.tqdm(frame_numbers):
            try:
                image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
            except Exception as e:
                self.logger.warning(f"Failed to extract frame {frame_number}: {e}")
                continue    
            
            prompt = self.get_prompt('default', parameters)
            prompts.append(prompt)
            images_batch.append(image)
            
            timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
            timepoint.add_property("timePoint", frame_number)
            annotations_batch.append({'source': timepoint.long_id})

            if len(prompts) == batch_size:
                start_time = time.time()
                self._process_batch(prompts, images_batch, annotations_batch, new_view)
                self.logger.info(f"Processed batch of {batch_size} frames in {time.time() - start_time:.2f} seconds")
                prompts, images_batch, annotations_batch = [], [], []

        # Process remaining frames
        if prompts:
            start_time = time.time()
            self._process_batch(prompts, images_batch, annotations_batch, new_view)
            self.logger.info(f"Processed final batch of {len(prompts)} frames in {time.time() - start_time:.2f} seconds")

    
def get_app():
    """Return an instance of the LlavaCaptioner app."""
    return LlavaCaptioner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = LlavaCaptioner()

    http_app = Restifier(app, port=int(parsed_args.port))

    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()