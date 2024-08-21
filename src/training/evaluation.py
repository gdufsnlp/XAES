from typing import List, Union

import numpy as np
from sklearn.metrics import cohen_kappa_score


def eval_fn_for_plms(
        trgs: np.ndarray, outs: np.ndarray, test_datasets: List[str], test_scalers,
        is_dev_mode=False,
) -> Union[dict, List[dict]]:
    """Evaluation function for pre-trained language models.

    Parameters:
        trgs: (np.ndarray) Ground-truth labels.
        outs: (np.ndarray) Outputs of the model.
        test_datasets: Names of the test datasets.
        test_scalers: Scalers for the test datasets.
        is_dev_mode: (bool) True for model validation and False for model inference.
    Return:
        When is_dev_mode is False (testing): [
            {
                "test_dataset": test_dataset,
                "agreement_measures": {
                    "qwk": qwk,
                    ...
                },
                "trgs": trgs,
                "outs": outs,
            }
        ]

        When is_dev_mode is True (validating): {
            "qwk": qwk,
            ...
        }
    """

    assert len(trgs) == len(outs) == len(test_datasets)

    """Trgs and outs grouped by test dataset: {
        test_dataset_1: {
            "trgs": trgs of the test dataset,
            "outs": outs of the test dataset,
        },
        test_dataset_2: {
            "trgs": trgs of the test dataset,
            "outs": outs of the test dataset,
        }
    }
    """
    eval_data = {}
    for trg, out, test_dataset in zip(trgs, outs, test_datasets):
        if test_dataset not in eval_data.keys():
            eval_data[test_dataset] = {
                "trgs": [],
                "outs": [],
            }
        eval_data[test_dataset]["trgs"].append(trg)
        eval_data[test_dataset]["outs"].append(out)

    eval_outputs = []
    for test_dataset, scores in eval_data.items():
        test_dataset_trgs = scores["trgs"]
        test_dataset_outs = scores["outs"]

        test_dataset_trgs = test_scalers[test_dataset].inverse_transform(
            np.array(test_dataset_trgs).reshape(-1, 1)
        ).flatten()
        test_dataset_outs = test_scalers[test_dataset].inverse_transform(
            np.array(test_dataset_outs).reshape(-1, 1)
        ).flatten()

        test_dataset_trgs = list(map(
            round,
            test_dataset_trgs,
        ))
        test_dataset_outs = list(map(
            round,
            test_dataset_outs,
        ))

        test_dataset_qwk = cohen_kappa_score(
            y1=test_dataset_trgs,
            y2=test_dataset_outs,
            weights="quadratic",
        )

        eval_outputs.append({
            "test_dataset": test_dataset,
            "agreement_measures": {
                "qwk": test_dataset_qwk,
            },
            "trgs": test_dataset_trgs,
            "outs": test_dataset_outs,
        })

    if is_dev_mode:  # TODO: improve the combing.
        # Combine the trgs and outs for dev evaluation.
        combined_trgs = []
        combined_outs = []
        for eval_output in eval_outputs:
            combined_trgs.extend(eval_output["trgs"])
            combined_outs.extend(eval_output["outs"])

        return {
            "qwk": cohen_kappa_score(
                combined_trgs,
                combined_outs,
                weights="quadratic",
            )
        }
    else:
        return eval_outputs
