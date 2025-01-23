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

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
        )
        super().__init__()

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, config: dict) -> str:
        if 'custom_prompts' in config and label in config['custom_prompts']:
            return config['custom_prompts'][label]
        return config['default_prompt']

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        print("config_file: ", config_file)
        config_dir = Path(__file__).parent
        config_file = config_dir / config_file
        config = self.load_config(config_file)
        
        batch_size = 8
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        def process_batch(prompts_batch, images_batch, annotations_batch):
            try:

                inputs = self.processor(images=images_batch, text=prompts_batch, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    # max_length=200,
                    max_new_tokens=200,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
                generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
                
                for generated_text, annotation in zip(generated_texts, annotations_batch):
                    text_document = new_view.new_textdocument(generated_text.strip())
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("source", annotation['source'])
                    alignment.add_property("target", text_document.long_id)
            finally:
                del inputs
                del outputs
                del generated_texts
                torch.cuda.empty_cache()

        input_context = config['context_config']['input_context']
        
        if input_context == "image":
            print("input_context: image")
            image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
            
            # Process images in batches
            for i in range(0, len(image_docs), batch_size):
                batch_docs = image_docs[i:i + batch_size]
                prompts = [config['default_prompt']] * len(batch_docs)
                # Load images using PIL
                images = [Image.open(doc.location_path()) for doc in batch_docs]
                annotations_batch = [{'source': doc.long_id} for doc in batch_docs]
                
                start_time = time.time()
                process_batch(prompts, images, annotations_batch)
                print(f"Processed batch of {len(batch_docs)} in {time.time() - start_time:.2f} seconds")
            
        elif input_context == 'timeframe':
            print("input_context: ", input_context)
            app_uri = config['context_config']['timeframe']['app_uri']
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            for view in all_views:
                print(view.metadata.app)
                if app_uri in view.metadata.app:
                    print("found view with app_uri: ", app_uri)
                    timeframes = view.get_annotations(AnnotationTypes.TimeFrame)
                    break
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
        elif input_context == 'fixed_window':
            print("input_context: ", input_context)
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]  # Get first video document
            window_duration = config['context_config']['fixed_window']['window_duration']
            stride = config['context_config']['fixed_window']['stride']
            fps = float(video_doc.get_property('fps'))
            total_frames = int(video_doc.get_property('frameCount'))
            frame_numbers = list(range(0, total_frames, int(fps * stride)))
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        if input_context == 'timeframe':
            # Convert timeframes generator to list
            timeframes = list(timeframes)
            # Get all middle frame numbers first
            frame_numbers = [vdh.get_mid_framenum(mmif, timeframe) for timeframe in timeframes]
            # Batch extract all images
            all_images = vdh.extract_frames_as_images(video_doc, frame_numbers, as_PIL=True)
            
            # Process in batches
            for i in tqdm.tqdm(range(0, len(timeframes), batch_size)):
                batch_timeframes = timeframes[i:i + batch_size]
                batch_images = all_images[i:i + batch_size]
                
                # Prepare batch data
                prompts = []
                annotations_batch = []
                for timeframe in batch_timeframes:
                    label = timeframe.get_property('label')
                    mapped_label = label_mapping.get(label, 'default')
                    prompt = self.get_prompt(mapped_label, config)
                    prompts.append(prompt)
                    annotations_batch.append({'source': timeframe.long_id})
                
                start_time = time.time()
                process_batch(prompts, batch_images, annotations_batch)
                print(f"Processed batch of {len(batch_timeframes)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'fixed_window':  # fixed_window
            prompts = []
            images_batch = []
            annotations_batch = []
            for frame_number in tqdm.tqdm(frame_numbers):
                image = vdh.extract_frame_as_image(video_doc, frame_number, as_PIL=True)
                prompt = config['default_prompt']
                prompts.append(prompt)
                images_batch.append(image)
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations_batch.append({'source': timepoint.long_id})

                if len(prompts) == batch_size:
                    start_time = time.time()
                    process_batch(prompts, images_batch, annotations_batch)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Processed a batch of {batch_size} in {elapsed_time:.2f} seconds.")
                    prompts, images_batch, annotations_batch = [], [], []

            if prompts:
                start_time = time.time()
                process_batch(prompts, images_batch, annotations_batch)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Processed the final batch of {len(prompts)} in {elapsed_time:.2f} seconds.")

        return mmif

    
def get_app():
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
