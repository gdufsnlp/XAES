import csv
import random
from collections import Counter
from typing import Dict, Tuple
from typing import List


class DataPreprocessor:
    required_fields = [
        "essay_id",
        "essay_set",
        "essay",
        "essay_score",
    ]

    int2text_cefr_mapping = {
        1: "A1",
        2: "A2",
        3: "B1",
        4: "B2",
        5: "C1",
        6: "C2",
    }

    """Data preprocessor.

    :param lang: (str) Language of the dataset.
    :param dataset: (str) Name of the dataset.
    :param file_type: (str) File type such as json and txt (if any).
    """

    def __init__(self, lang: str, dataset: str, file_type: str = None):
        self.lang = lang
        self.dataset = dataset
        self.file_type = file_type

        self.examples = self.read()  # Read and preprocess essays.

    def read_examples(self) -> List[str]:
        """Read examples from files."""
        raise NotImplementedError

    def preprocess(self, example: str) -> Dict:
        """Preprocess each example.

        :param example: (str) An example in the dataset. It
            may be a string, a json object or other types.
        :return: (dict) The preprocessed example.
        """
        raise NotImplementedError

    def read(self):
        """Read and preprocess examples from the dataset."""

        examples = list(filter(
            bool,
            list(map(
                self.preprocess,
                self.read_examples(),
            ))
        ))
        # Shuffle the examples.
        random.shuffle(examples)

        # Update field names of the required fields.
        for example in examples:
            for k, v in self.required_field_mapping.items():
                example.update({
                    v: example.pop(k)
                })

        return examples

    def to_tsv(self):
        """Write the examples into a tsv file."""

        fieldnames = self.required_fields + sorted(list(
            set(self.examples[0].keys()) - set(self.required_fields)
        ))
        with open(self.default_tsvp, 'w', encoding="utf-8") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.examples)

    @property
    def score_statistics(self):
        """Print score statistics."""
        s = ""
        for int_score, text_score in self.int2text_score_mapping.items():
            s += f"  {text_score} ({int_score}): "
            s += str(len([
                example
                for example in self.examples
                if str(example["essay_score"]) == str(int_score)
            ]))
            s += "\n"
        return s.rstrip()

    @property
    def theme_statistics(self):
        """Print theme statistics."""
        s = ""
        theme_counter = Counter(list(map(
            lambda example: str(example["essay_set"]),
            self.examples,
        )))
        for i, theme in enumerate(self.themes):
            s += f"  {i + 1}. {theme}: {theme_counter[theme]}"
            s += "\n"
        return s.rstrip()

    @property
    def default_fp(self) -> str:
        """Default path of the data file (if any)."""
        return f"data/ori/{self.lang}/{self.dataset}/{self.dataset}.{self.file_type}"

    @property
    def default_dp(self) -> str:
        """Default path of the data dir (if any)."""
        return f"data/ori/{self.lang}/{self.dataset}"

    @property
    def default_tsvp(self) -> str:
        """Default path of the tsv file."""
        return f"data/raw/{self.lang}_{self.dataset}.tsv"

    @property
    def required_field_mapping(self) -> Dict:
        """original_field_name: required_field_name."""
        raise NotImplementedError

    @property
    def int2text_score_mapping(self) -> Dict:
        """int_score: textual_score_description.

        E.g., {
            1: "A1",
            2: "A2",
            3: "B1",
            4: "B2",
            5: "C1",
            6: "C2",
        }
        """
        raise NotImplementedError

    @property
    def text2int_score_mapping(self) -> Dict:
        """textual_score_description: int_score.

        E.g., {
            "A1": 1,
            "A2": 2,
            "B1": 3,
            "B2": 4,
            "C1": 5,
            "C2": 6,
        }
        """
        return {
            v: k
            for k, v in self.int2text_score_mapping.items()
        }

    @property
    def themes(self) -> Tuple:
        """Themes (essay sets)."""
        return tuple(sorted(set(list(map(
            lambda example: str(example["essay_set"]).strip(),
            self.examples,
        )))))

    def __repr__(self):
        s = ""

        s += f"#Examples: {len(self.examples)}" + "\n"
        s += f"#Scores: {len(self.int2text_score_mapping)}" + "\n"
        s += self.score_statistics + "\n"
        s += f"#Themes: {len(self.themes)}" + "\n"
        s += self.theme_statistics + "\n"

        return s.strip()
