import argparse
import logging

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor
import torch


class LlavaCaptioner(ClamsApp):

    def __init__(self):
        super().__init__()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            quantization_config=quantization_config, 
            device_map="auto"
        )
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def get_prompt(self, label: str, prompt_map: dict, default_prompt: str) -> str:
        prompt = prompt_map.get(label, default_prompt)
        if prompt == "-":
            return None
        prompt = f"[INST] <image>\n{prompt}\n[/INST]"
        return prompt

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        label_map = parameters.get('promptMap')
        default_prompt = parameters.get('defaultPrompt')
        frame_interval = parameters.get('frameInterval', 10)  # Default to every 10th frame if not specified

        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        
        timeframes = input_view.get_annotations(AnnotationTypes.TimeFrame)
        
        if timeframes:
            for timeframe in timeframes:
                label = timeframe.get_property('label')
                prompt = self.get_prompt(label, label_map, default_prompt)
                if not prompt:
                    continue

                representatives = timeframe.get("representatives") if "representatives" in timeframe.properties else None
                if representatives:
                    image = vdh.extract_representative_frame(mmif, timeframe)
                else:
                    image = vdh.extract_mid_frame(mmif, timeframe)
                # clear gpu cache
                torch.cuda.empty_cache()
                inputs = self.processor(prompt, image, return_tensors="pt")
                output = self.model.generate(**inputs, max_new_tokens=100)
                description = self.processor.decode(output[0], skip_special_tokens=True)
                print (description)
                text_document = new_view.new_textdocument(description)
                # todo discuss this alignment missing timepoint information
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", timeframe.id)
                alignment.add_property("target", text_document.id)
        else:
            total_frames = vdh.get_frame_count(video_doc)
            for frame_number in range(0, total_frames, frame_interval):
                image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
                prompt = default_prompt
                inputs = self.processor(prompt, image, return_tensors="pt")
                output = self.model.generate(**inputs, max_new_tokens=250)
                description = self.processor.decode(output[0], skip_special_tokens=True)

                text_document = new_view.new_textdocument(description)
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", timepoint.id)
                alignment.add_property("target", text_document.id)

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument("--frameInterval", type=int, default=10, help="Interval of frames for captioning when no timeframes are present")

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
