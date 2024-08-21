import os
from typing import Dict

from src.data.preprocessors.data_preprocessor import DataPreprocessor


class SpanishCedel2DataPreprocessor(DataPreprocessor):
    raw2cefr_score_mapping = {
        range(0, 12 + 1): "A1",
        range(13, 20 + 1): "A2",
        range(21, 28 + 1): "B1",
        range(29, 35 + 1): "B2",
        range(36, 40 + 1): "C1",
        range(41, 43 + 1): "C2"
    }

    def __init__(self):
        super(SpanishCedel2DataPreprocessor, self).__init__(
            lang="es",
            dataset="cedel2",
        )

    def read_examples(self):
        examples = []
        for file in sorted(os.listdir(self.default_dp)):
            with open(os.path.join(self.default_dp, file), "r", encoding="utf-8") as f:
                examples.append(f.read())
        return examples

    def preprocess(self, example: str):
        """http://cedel2.learnercorpora.com/statistics"""

        lines = example.split("\n")

        example = {}
        is_making_text = False  # True when reaching the line starting with "Text:".
        for line in lines:
            # Lines after the line starting with "Text:" belong to the text
            # the writer wrote.
            if is_making_text is False and line.startswith("Text:"):
                is_making_text = True
                example["Text"] = ""

            # Make text.
            if is_making_text:
                example["Text"] += "\n" + line
            # Attrs.
            else:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()

                example[k] = v

        # Replace the first "Text: "
        example["Text"] = example["Text"].replace("Text: ", "", 1)
        # Strip the text.
        example["Text"] = example["Text"].strip()

        # Add CEFR level.
        raw_score = int(example["Placement test score (raw)"].split("/", 1)[0])
        for raw_score_range, cefr_level in self.raw2cefr_score_mapping.items():
            if raw_score not in raw_score_range:
                continue

            example["CEFR"] = cefr_level
            # Convert textual score to int.
            example["int_cefr"] = self.text2int_score_mapping[example["CEFR"]]

        # Only keep writings.
        if example["Medium"] == "Written":
            return example
        else:
            return None

    @property
    def int2text_score_mapping(self):
        return super().int2text_cefr_mapping

    @property
    def required_field_mapping(self) -> Dict:
        return {
            "Filename": "essay_id",
            "Task title": "essay_set",
            "Text": "essay",
            "int_cefr": "essay_score",
        }


if __name__ == '__main__':
    SpanishCedel2DataPreprocessor().to_tsv()
