import argparse
import logging
import yaml
from pathlib import Path

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor
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
            attn_implementation="flash_attention_2"
        )
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
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
        config_file = parameters.get('config', 'config/shot_captioning.yaml')
        # get containing directory of current file
        config_dir = Path(__file__).parent
        # get absolute path
        config_file = config_dir / config_file
        config = self.load_config(config_file)
        
        # batch_size = parameters.get('batchSize', 4)  # Default batch size
        batch_size = 2
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        # input_view: View = mmif.get_views_for_document(video_doc.id)[-1]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        input_context = config['context_config']['input_context']
        
        if input_context == 'timeframe':
            app_uri = config['context_config']['timeframe']['app_uri']
            # get all views with timeframe annotations and iterate over them to find the one with the correct app_id
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            for view in all_views:
                if app_uri in view.metadata.app:
                    timeframes = view.get_annotations(AnnotationTypes.TimeFrame)
                    break
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
        elif input_context == 'fixed_window':
            window_duration = config['context_config']['fixed_window']['window_duration']
            stride = config['context_config']['fixed_window']['stride']
            fps = float(video_doc.get_property('fps'))
            total_frames = int(video_doc.get_property('frameCount'))
            frame_numbers = list(range(0, total_frames, int(fps * stride)))
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        prompts = []
        images = []
        annotations = []

        def process_batch(prompts, images, annotations):
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=200,
                min_length=1,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # NOTE: when creating a new caption text document from how should we align it to the given timeframe?
            # it is running on one frame, so it makes sense to align it to a timepoint, but the timeframe is providing context
            # for the caption and potentially determining the prompt.
            # We could create 2 alignments, but that seems messy.
            # We could create a single alignment from the text document to the timeframe and use the properties to specify which frame was used?
            for generated_text, annotation in zip(generated_texts, annotations):
                print ("generated_text: ", generated_text)
                text_document = new_view.new_textdocument(generated_text.strip())
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", annotation['source'])
                alignment.add_property("target", text_document.long_id)

        if input_context == 'timeframe':
            for timeframe in timeframes:
                label = timeframe.get_property('label')
                mapped_label = label_mapping.get(label, 'default')
                
                prompt = self.get_prompt(mapped_label, config)

                image = vdh.extract_mid_frame(mmif, timeframe, as_PIL=True)
                prompts.append(prompt)
                images.append(image)
                annotations.append({'source': timeframe.long_id})

                if len(prompts) == batch_size:
                    process_batch(prompts, images, annotations)
                    prompts, images, annotations = [], [], []
        else:  # fixed_window
            images = vdh.extract_frames_as_images(video_doc, frame_numbers, as_PIL=True)
            for frame_number, image in zip(frame_numbers, images):
                prompt = config['default_prompt']
                prompts.append(prompt)
                images.append(image)
                # Create new timepoint annotation
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations.append({'source': timepoint.long_id})

                if len(prompts) == batch_size:
                    process_batch(prompts, images, annotations)
                    prompts, images, annotations = [], [], []

        if prompts:
            process_batch(prompts, images, annotations)

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
