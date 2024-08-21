import os
from typing import Dict

from src.data.preprocessors.data_preprocessor import DataPreprocessor


class MerlinDataPreprocessor(DataPreprocessor):

    def __init__(self, lang: str, dataset: str):
        super(MerlinDataPreprocessor, self).__init__(
            lang=lang,
            dataset=dataset,
        )

    def read_examples(self):
        examples = []
        for file in sorted(os.listdir(self.default_dp)):
            with open(os.path.join(self.default_dp, file), "r", encoding="utf-8") as f:
                examples.append(f.read())
        return examples

    def preprocess(self, example: str):
        lines = example.split("\n")

        example = {}
        is_making_text = False  # True when reaching "Learner text:"
        for line in lines:
            # Ignore useless info.
            if line.strip() in [
                "",
                "METADATA",
                "General:",
                "Rating:",
                "----------------"
            ]:
                continue
            if line.startswith("Notice: Undefined index:"):
                continue

            # Lines after "Learner text:" belong to the text
            # the writer wrote.
            if is_making_text is False and line.strip() == "Learner text:":
                is_making_text = True
                example["Learner text"] = ""
                continue

            # Make text.
            if is_making_text:
                example["Learner text"] += line + "\n"
            # Meta data.
            else:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()

                example[k] = v

        # Strip the learner text.
        example["Learner text"] = example["Learner text"].strip()

        # Ignore unrated essays.
        if example["Overall CEFR rating"] in ["unrated", "EMPTY"]:
            return None
        else:
            # Convert textual score to int.
            example["int_cefr"] = self.text2int_score_mapping[example["Overall CEFR rating"]]
            return example

    @property
    def int2text_score_mapping(self):
        return super().int2text_cefr_mapping

    @property
    def required_field_mapping(self) -> Dict:
        return {
            "Author ID": "essay_id",
            "Task": "essay_set",
            "Learner text": "essay",
            "int_cefr": "essay_score",
        }
