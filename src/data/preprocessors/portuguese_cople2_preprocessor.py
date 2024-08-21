import csv
from typing import Dict

from src.data.preprocessors.data_preprocessor import DataPreprocessor


class PortugueseCople2DataPreprocessor(DataPreprocessor):

    def __init__(self):
        super(PortugueseCople2DataPreprocessor, self).__init__(
            lang="pt",
            dataset="cople2",
            file_type="tsv",
        )

    def read_examples(self):
        examples = []
        with open(self.default_fp, encoding="utf-8") as f:
            dict_reader = csv.DictReader(f, delimiter="\t")
            for example in dict_reader:
                examples.append(example)
        return examples

    def preprocess(self, example: dict):
        if example["Genre"] == "dialogue":
            # Not a written text.
            return None

        if example["text"].strip() == "":
            # No text.
            return None

        example["ID"] = example["ID"].replace(".xml", "")

        try:
            example["Proficiency"] = self.text2int_score_mapping[example["Proficiency"]]
        except KeyError:
            # No score provided.
            return None

        return example

    @property
    def int2text_score_mapping(self):
        return super().int2text_cefr_mapping

    @property
    def required_field_mapping(self) -> Dict:
        return {
            "ID": "essay_id",
            "Prompt": "essay_set",
            "text": "essay",
            "Proficiency": "essay_score",
        }


if __name__ == '__main__':
    PortugueseCople2DataPreprocessor().to_tsv()
