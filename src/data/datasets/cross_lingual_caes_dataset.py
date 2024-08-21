import csv
import os
from typing import List, Callable, Type

from sklearn.model_selection import train_test_split

SCORE_RANGES = {
    "cs_merlin": range(1, 6 + 1),
    "de_merlin": range(1, 6 + 1),
    "en_write_and_improve": range(1, 6 + 1),
    "es_cedel2": range(1, 6 + 1),
    "it_merlin": range(1, 6 + 1),
    "pt_cople2": range(1, 6 + 1),

    # TODO: for debugging.
    ".cs_merlin.debug": range(1, 6 + 1),
    ".de_merlin.debug": range(1, 6 + 1),
    ".en_write_and_improve.debug": range(1, 6 + 1),
    ".es_cedel2.debug": range(1, 6 + 1),
    ".it_merlin.debug": range(1, 6 + 1),
}


class CrossLingualAESDataset:
    """Dataset for cross-lingual automated essay scoring.

    Parameters:
        dp_datasets: (str) Directory that contains the tsv files of the datasets.
        source_datasets: (List[str]) Datasets for model training.
        target_datasets: (List[str]) Datasets for model testing.
        example_merging_fn: (Callable) Function for merging examples.
        example_merging_config: (dict) Config dict for merging examples.
        essay_processing_fn: (Callable) Function for processing essays.
        essay_processing_config: (dict) Config dict for processing essays.
        dev_ratio: (float) Ratio of the development set.
        data_split_cls: (Type): Class for the data split.
        data_split_config: (dict) Config dict for data splitting.
        n_each_source_lang: (int) Number of examples for each source language.
    """

    def __init__(
            self,
            dp_datasets: str, source_datasets: List[str], target_datasets: List[str],
            example_merging_fn: Callable, example_merging_config: dict,
            essay_processing_fn: Callable, essay_processing_config: dict,
            dev_ratio: float,
            data_split_cls: Type, data_split_config: dict,
            n_each_source_lang: int
    ):
        self.dp_datasets = dp_datasets
        self.source_datasets = source_datasets
        self.target_datasets = target_datasets

        self.example_merging_fn = example_merging_fn
        self.essay_processing_fn = essay_processing_fn
        self.configs = {
            "example_merging": example_merging_config,
            "essay_processing": essay_processing_config,
        }

        self.dev_ratio = dev_ratio

        self.data_split_cls = data_split_cls
        self.data_split_config = data_split_config  # TODO: merge to self.configs.

        self.n_each_source_lang = n_each_source_lang

        (
            self.train, self.dev, self.test,
            self.train_datasets, self.dev_datasets, self.test_datasets,  # Names of the datasets.
            self.train_ids, self.dev_ids, self.test_ids,
        ) = self.make()

    def make(self):
        print("Reading examples.")
        source_examples = self.read_examples(datasets=self.source_datasets)
        target_examples = self.read_examples(datasets=self.target_datasets)

        print("Merging examples.")
        source_examples = self.example_merging_fn(
            examples=source_examples,
            configs=self.configs
        )
        target_examples = self.example_merging_fn(
            examples=target_examples,
            configs=self.configs,
        )

        if self.n_each_source_lang is not None:
            print(f"Sampling {self.n_each_source_lang} essays for each source language.")
            source_examples = self.sample(
                examples=source_examples,
                datasets=self.source_datasets,
            )

        print("Train/dev splitting.")
        train_examples, dev_examples = train_test_split(
            source_examples,
            test_size=self.dev_ratio,
        )
        test_examples = target_examples
        print(f"#train: {len(train_examples)}")
        print(f"#dev:   {len(dev_examples)}")
        print(f"#test:  {len(test_examples)}")

        train_datasets, train_ids, train_essays, train_scores = self.extract_components(examples=train_examples)
        dev_datasets, dev_ids, dev_essays, dev_scores = self.extract_components(examples=dev_examples)
        test_datasets, test_ids, test_essays, test_scores = self.extract_components(examples=test_examples)

        train_essays = self.essay_processing_fn(
            essays=train_essays,
            configs=self.configs,
        )
        dev_essays = self.essay_processing_fn(
            essays=dev_essays,
            configs=self.configs,
        )
        test_essays = self.essay_processing_fn(
            essays=test_essays,
            configs=self.configs,
        )

        train = self.data_split_cls(
            split="train",
            xs=train_essays,
            ys=train_scores,
            kwargs=self.data_split_config,
        )
        dev = self.data_split_cls(
            split="dev",
            xs=dev_essays,
            ys=dev_scores,
            kwargs=self.data_split_config,
        )
        test = self.data_split_cls(
            split="test",
            xs=test_essays,
            ys=test_scores,
            kwargs=self.data_split_config,
        )

        return (
            train, dev, test,
            train_datasets, dev_datasets, test_datasets,
            train_ids, dev_ids, test_ids,
        )

    def sample(self, examples, datasets):
        dataset2count = {
            dataset: 0
            for dataset in datasets
        }

        sampled_examples = []
        for example in examples:
            if dataset2count[example["dataset"]] < self.n_each_source_lang:
                sampled_examples.append(example)

                dataset2count[example["dataset"]] += 1

        return sampled_examples

    def read_examples(self, datasets: List[str]) -> dict:
        """Read examples from the given datasets.

        Parameters:
            datasets: (List[str]) Names of the datasets.
        Return:
            dict {
                dataset1: [
                    {
                        "essay": essay1,
                        "score": score1
                    },
                    {
                        "essay": essay2,
                        "score": score2,
                    },
                    ...
                ],
                dataset2: [
                    {
                        "essay": essay1,
                        "score": score1
                    },
                    {
                        "essay": essay2,
                        "score": score2,
                    },
                    ...
                ],
                ...
            }
        """

        examples = {}
        for dataset in datasets:
            examples[dataset] = []

            fp_dataset = os.path.join(
                self.dp_datasets,
                f"{dataset}.tsv"
            )
            with open(fp_dataset, "r", encoding="utf-8") as f:
                tsv_reader = csv.DictReader(f, delimiter="\t")
                for example in tsv_reader:
                    examples[dataset].append({
                        "id": example["essay_id"],
                        "essay": example["essay"],
                        "score": float(example["essay_score"]),
                    })

            print(f"#{dataset}: {len(examples[dataset]):,}")

        return examples

    @staticmethod
    def extract_components(examples):
        """Obtain datasets, essays, scores from examples."""
        datasets = []
        ids = []
        essays = []
        scores = []
        for example in examples:
            datasets.append(example["dataset"])
            ids.append(example["id"])
            essays.append(example["essay"])
            scores.append(example["score"])

        return datasets, ids, essays, scores
