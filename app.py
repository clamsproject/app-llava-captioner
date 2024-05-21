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
    
    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            representatives = timeframe.get("representatives") if "representatives" in timeframe.properties else None
            if representatives:
                representative = input_view.get_annotation_by_id(representatives[0])
                rep_frame = vdh.convert(representative.get("timePoint"), "milliseconds", "frame", vdh.get_framerate(video_doc))
            else:
                start_time = timeframe.get("start")
                end_time = timeframe.get("end")
                rep_frame = (start_time + end_time) / 2
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property('timePoint', rep_frame)

            image = vdh.extract_frames_as_images(video_doc, [rep_frame], as_PIL=True)[0]

            prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
            inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")
            output = self.model.generate(**inputs, max_new_tokens=100)
            description = self.processor.decode(output[0], skip_special_tokens=True)

            print(description)
            text_document = new_view.new_textdocument(description)
            alignment = new_view.new_annotation(AnnotationTypes.Alignment)
            alignment.add_property("source", timeframe.id)
            # todo, if choosing the middle frame the alignment should be to that frame?
            alignment.add_property("target", text_document.id)

        return mmif


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
