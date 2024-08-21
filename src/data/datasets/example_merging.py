import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data.datasets.cross_lingual_caes_dataset import SCORE_RANGES


def example_merging_fn_for_plms(examples: dict, configs: dict):
    """Merging function for pre-trained language models.

    Parameters:
        examples: (dict) List of {
            "essay": essay,
            "score": score,
        } grouped by dataset.
        configs: (dict) Config dict.
    Return:
        (examples, configs).
        examples: List of {
            "dataset": dataset,
            "essay": essay,
            "score": score,
        }

    Essays are directly merged. Scores are scaled and merged.
    """

    merged_examples = []
    configs["example_merging"].setdefault("score_scalers", {})
    for dataset, dataset_examples in examples.items():
        dataset_ids = []
        dataset_essays = []
        dataset_scores = []
        for dataset_example in dataset_examples:
            dataset_ids.append(dataset_example["id"])
            dataset_essays.append(dataset_example["essay"])
            dataset_scores.append(dataset_example["score"])

        dataset_scaler = MinMaxScaler()
        dataset_scaler.fit(
            np.array(SCORE_RANGES[dataset]).reshape(-1, 1),
        )
        dataset_scores = dataset_scaler.transform(
            np.array(dataset_scores).reshape(-1, 1),
        ).flatten()

        for dataset_id, dataset_essay, dataset_score in zip(dataset_ids, dataset_essays, dataset_scores):
            merged_examples.append({
                "dataset": dataset,
                "id": dataset_id,
                "essay": dataset_essay,
                "score": dataset_score,
            })
        configs["example_merging"]["score_scalers"][dataset] = dataset_scaler

    # Shuffle the merged examples.
    random.shuffle(merged_examples)

    print(f"#{','.join(list(examples.keys()))}: {len(merged_examples):,}")

    return merged_examples
