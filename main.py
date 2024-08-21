import argparse
import json
import os.path
import os.path
from typing import Union

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)

from src.data.datasets.cross_lingual_caes_dataset import CrossLingualAESDataset
from src.data.datasets.data_split import DataSplitForPLMs
from src.data.datasets.essay_processing import essay_processing_fn_for_plms
from src.data.datasets.example_merging import example_merging_fn_for_plms
from src.models.bert_for_sequence_classification_with_supcl import BertForSequenceClassificationWithSupCL
from src.training.evaluation import eval_fn_for_plms
from src.utils import fix_seed, get_git_revision_hash

Arch2PLM = {
    "bert-base-multilingual-uncased": "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert-uncased-for-sequence-classification-with-supcl": "bert-base-multilingual-uncased",
}

PLM2ModelClass = {
    "bert-base-multilingual-uncased": AutoModelForSequenceClassification,
    "bert-base-multilingual-cased": AutoModelForSequenceClassification,
    "xlm-roberta-base": AutoModelForSequenceClassification,
    "mbert-uncased-for-sequence-classification-with-supcl": BertForSequenceClassificationWithSupCL,
}

PLMsForSequenceClassificationWithSupCL = (
    "mbert-uncased-for-sequence-classification-with-supcl",
)


def evaluate(
        output_wrapper: Union[EvalPrediction, PredictionOutput], dataset: CrossLingualAESDataset,
        is_dev_mode: bool = False,
) -> Union[list, dict]:
    trgs = output_wrapper.label_ids.flatten()
    outs = output_wrapper.predictions.flatten()

    # Automatically infer the split.
    if len(dataset.dev_datasets) == len(trgs):
        eval_datasets = dataset.dev_datasets
        print("Evaluating for dev.")
    elif len(dataset.test_datasets) == len(trgs):
        eval_datasets = dataset.test_datasets
        print("Evaluating for test.")
    else:
        raise NotImplementedError

    eval_scalers = dataset.configs["example_merging"]["score_scalers"]

    return eval_fn_for_plms(
        trgs=trgs,
        outs=outs,
        test_datasets=eval_datasets,
        test_scalers=eval_scalers,
        is_dev_mode=is_dev_mode,
    )


def _get_pretrained_model_name_or_path(args):
    return (
        Arch2PLM[args.plm_arch]
        if args.plm_restore_ckpt is None
        else args.plm_restore_ckpt
    )


def load_tokenizer(args):
    pretrained_model_name_or_path = _get_pretrained_model_name_or_path(args)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        model_max_length=args.max_seq_len,
    )
    print(f"Tokenizer: {tokenizer.__class__.__name__} loaded from {pretrained_model_name_or_path}")

    return tokenizer


def load_config(args):
    pretrained_model_name_or_path = _get_pretrained_model_name_or_path(args)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        num_labels=1,
    )
    print(f"Config: {config.__class__.__name__} loaded from {pretrained_model_name_or_path}")

    return config


def load_model(args, config):
    pretrained_model_name_or_path = _get_pretrained_model_name_or_path(args)
    model = PLM2ModelClass[args.plm_arch].from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config,
    )
    print(f"Model: {model.__class__.__name__} loaded from {pretrained_model_name_or_path}")

    return model


def main(args: argparse.Namespace):
    tokenizer = load_tokenizer(args)
    dataset = CrossLingualAESDataset(
        dp_datasets=args.dp_datasets,
        source_datasets=args.source_datasets,
        target_datasets=args.target_datasets,
        example_merging_fn=example_merging_fn_for_plms,
        example_merging_config={},
        essay_processing_fn=essay_processing_fn_for_plms,
        essay_processing_config={
            "tokenization": {
                "tokenizer": tokenizer,
                "args": {
                    "truncation": True,
                    "padding": True,
                },
                "lowercase": args.lowercase
            }
        },
        data_split_cls=DataSplitForPLMs,
        data_split_config=dict(
            include_writing_qualities=args.plm_arch in PLMsForSequenceClassificationWithSupCL,
            writing_quality_granularity=args.writing_quality_granularity,
        ),
        dev_ratio=args.dev_ratio,
        n_each_source_lang=args.n_each_source_lang,
    )

    config = load_config(args)
    config.sup_cl = {
        "cl_temp": args.cl_temp,
        "mse_weight": args.mse_weight,
        "cl_weight": args.cl_weight,
        "cl_memory_bank_size": args.cl_memory_bank_size,
        "cl_projector": args.cl_projector,
    }
    model = load_model(
        args=args,
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.dp_outputs,
            seed=args.seed,

            learning_rate=args.learning_rate,

            per_device_train_batch_size=args.train_bz,
            per_device_eval_batch_size=args.eval_bz,
            gradient_accumulation_steps=args.train_accum,
            eval_accumulation_steps=args.eval_accum,

            num_train_epochs=args.n_epochs,

            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=args.save_total_limit,

            load_best_model_at_end=True,
            metric_for_best_model="qwk",
            greater_is_better=True,
        ),
        train_dataset=dataset.train,
        eval_dataset=dataset.dev,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        compute_metrics=lambda eval_prediction: evaluate(
            output_wrapper=eval_prediction,
            dataset=dataset,
            is_dev_mode=True,
        )
    )
    if not args.no_train:
        trainer.train()
        trainer.save_model(output_dir=os.path.join(
            args.dp_outputs,
            "checkpoint-best",
        ))

    model.config.return_dict = True
    eval_output = evaluate(
        output_wrapper=trainer.predict(dataset.test),
        dataset=dataset,
        is_dev_mode=False,
    )

    return eval_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args.
    parser.add_argument(
        "--dp-datasets", required=True, type=str,
        help="Directory that contains the tsv files of the datasets.",
    )
    parser.add_argument(
        "--source-datasets", required=True, nargs="+",
        help="Datasets for training the model.",
    )
    parser.add_argument(
        "--target-datasets", required=True, nargs="+",
        help="Datasets for testing the model.",
    )
    parser.add_argument(
        "--dev-ratio", default=0.25, type=float,
        help="Ratio of the development set, 0.15 by default.",
    )
    parser.add_argument(
        "--n-each-source-lang", default=None, type=int,
        help="Number of essays for each source language."
    )

    # Tokenization args.
    parser.add_argument(
        "--lowercase", action="store_true", default=False,
        help="Lowercase texts."
    )
    parser.add_argument(
        "--max-seq-len", default=None, type=int,
        help="Max seq len of the model."
    )

    # Model args.
    parser.add_argument(
        "--plm-arch", type=str,
        help="Architecture of the pre-trained language model (e.g., bert-base-multilingual-uncased)."
    )
    parser.add_argument(
        "--plm-restore-ckpt", type=str,
        help="Checkpoint dir of the plm for restoring."
    )

    # Training args.
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed. 42 by default.",
    )
    parser.add_argument(
        "--learning-rate", default=5e-5, type=float,
        help="Learning rate. 5e-5 by default."
    )
    parser.add_argument(
        "--train-bz", default=8, type=int,
        help="Training batch size. 8 by default."
    )
    parser.add_argument(
        "--eval-bz", default=32, type=int,
        help="Evaluation batch size. 32 by default."
    )
    parser.add_argument(
        "--train-accum", default=2, type=int,
        help="Training accumulation steps. 2 by default."
    )
    parser.add_argument(
        "--eval-accum", default=2, type=int,
        help="Evaluation accumulation steps. 2 by default."
    )
    parser.add_argument(
        "--n-epochs", default=50, type=int,
        help="Number of epochs. 50 by default."
    )
    parser.add_argument(
        "--patience", default=5, type=int,
        help="Training patience. 5 by default."
    )
    parser.add_argument(
        "--dp-outputs", required=True, type=str,
        help="Path of the output directory."
    )
    parser.add_argument(
        "--save-total-limit", default=1, type=int,
        help="Max number of saved checkpoints. 1 by default."
    )
    parser.add_argument(
        "--cl-temp", default=0.1, type=float,
        help="Temperature for the CL loss. 0.1 by default."
    )
    parser.add_argument(
        "--mse-weight", default=1.0, type=float,
        help="Weight of the MSE loss. 1.0 by default."
    )
    parser.add_argument(
        "--cl-weight", default=1.0, type=float,
        help="Weight of the CL loss. 1.0 by default."
    )
    parser.add_argument(
        "--cl-memory-bank-size", default=128, type=int,
        help="Memory bank size for the CL loss. 128 by default."
    )
    parser.add_argument(
        "--cl-projector", default="no", type=str, choices=["no", "linear", "mlp"],
        help="Projector for CL (no, linear, mlp). \"no\" by default."
    )
    parser.add_argument(
        "--no-train", action="store_true", default=False,
        help="Skip model training."
    )

    parser.add_argument(
        "--writing-quality-granularity", type=int, default=3,
        help="Number of writing quality levels. 3 by default (low, medium and high quality)."
    )

    args = parser.parse_args()
    print(args)

    fix_seed(seed=args.seed)

    outputs = main(args)

    output_dict = {
        "git": get_git_revision_hash(),
        "args": args.__dict__,
        # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
        "outputs": outputs,
    }
    fp_output_json = os.path.join(args.dp_outputs, "output.json")
    with open(fp_output_json, "w", encoding="utf-8") as f:
        json.dump(
            obj=output_dict,
            fp=f,
            ensure_ascii=False,
            indent=2,
        )
