# XAES
Code for Paper: *[Zero-shot Cross-lingual Automated Essay Scoring](https://aclanthology.org/2024.lrec-main.1550/)*.

## Datasets
Follow the instructions in [data/README.md](data/README.md) to download and prepare AES datasets for six different languages.

## Usage
```bash
# Create a virtual environment and install necessary packages.
bash setup.sh
# Activate the virtual environment.
source .venv/bin/activate

# Preprocess data.
python3 preprocess_data.py

# Train a model and evaluate its performance.
# The training logs will be placed in `logs`.
# The checkpoints and results (including qwks) will be placed in `outs`.
bash scripts/run.sh
```

## Citation
```bibtex
@inproceedings{he-li-2024-zero,
    title = "Zero-shot Cross-lingual Automated Essay Scoring",
    author = "He, Junyi  and
      Li, Xia",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1550",
    pages = "17819--17832",
}
```