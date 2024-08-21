import json
import os
from typing import Dict

from src.data.preprocessors.data_preprocessor import DataPreprocessor


class EnglishWriteAndImproveDataPreprocessor(DataPreprocessor):

    def __init__(self):
        super(EnglishWriteAndImproveDataPreprocessor, self).__init__(
            lang="en",
            dataset="write_and_improve",
        )

    def read_examples(self):
        examples = []
        for file in sorted(os.listdir(self.default_dp)):
            with open(os.path.join(self.default_dp, file), "r", encoding="utf-8") as f:
                examples.extend(f.read().splitlines())
        return examples

    def preprocess(self, example: str):
        example = json.loads(example)

        # {A|B|C}.* -> {A|B|C}
        # C2+ -> C2
        example["cefr"] = example["cefr"].split(".")[0].replace("C2+", "C2")

        # Convert textual score to int.
        example["int_cefr"] = self.text2int_score_mapping[example["cefr"]]

        # Different files may have different k: v orders
        # in this dataset, thus sorting is needed.
        # Otherwise, values are in wrong orders when invoking
        # self.to_tsv().
        example = dict(sorted(
            example.items(),
            key=lambda item: item[0]
        ))

        # Add a pseudo essay set field.
        example["essay_set"] = "Unknown"

        return example

    @property
    def int2text_score_mapping(self):
        return super().int2text_cefr_mapping

    @property
    def required_field_mapping(self) -> Dict:
        return {
            "id": "essay_id",
            "text": "essay",
            "int_cefr": "essay_score",
        }


if __name__ == '__main__':
    EnglishWriteAndImproveDataPreprocessor().to_tsv()
