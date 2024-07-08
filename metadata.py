"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Llava Captioner",
        description="Applies llava to video frames.",
        app_license="Apache 2.0",
        identifier="llava-captioner",
        url="https://github.com/clamsproject/app-llava-captioner"
    )

    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.TimeFrame)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(DocumentTypes.TextDocument)
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(
        name='defaultPrompt', type='string', default='Describe what is shown in this video frame.',
        description='default prompt to use for timeframes not specified in the promptMap. If set to `-`, '
                     'timeframes not specified in the promptMap will be skipped.'
    )
    metadata.add_parameter(
        name='promptMap', type='map', default=[],
        description=('mapping of labels of the input timeframe annotations to new prompts. Must be formatted as '
                     '\"IN_LABEL:PROMPT\" (with a colon). To pass multiple mappings, use this parameter multiple '
                     'times. By default, any timeframe labels not mapped to a prompt will be used with the default'
                     'prompt. In order to skip timeframes with a particular label, pass `-` as the prompt value.'
                     'in order to skip all timeframes not specified in the promptMap, set the defaultPrompt'
                     'parameter to `-`'))
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
