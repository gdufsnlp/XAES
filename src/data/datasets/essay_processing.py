from typing import List


def essay_processing_fn_for_plms(essays: List[str], configs: dict):
    """Essay processing function for pre-trained language models.

    Parameters:
        essays: (List[str]): Essays to process.
        configs: (dict): Config dict.
    Return:
        Processed essays.
    """

    if configs["essay_processing"]["tokenization"]["lowercase"]:
        essays = list(map(
            lambda essay: essay.lower(),
            essays,
        ))

    return configs["essay_processing"]["tokenization"]["tokenizer"](
        essays,
        **configs["essay_processing"]["tokenization"]["args"]
    )
