import argparse
import os

from src.data.preprocessors.czech_merlin_data_preprocessor import CzechMerlinDataPreprocessor
from src.data.preprocessors.english_write_and_improve_data_preprocessor import EnglishWriteAndImproveDataPreprocessor
from src.data.preprocessors.german_merlin_data_preprocessor import GermanMerlinDataPreprocessor
from src.data.preprocessors.italian_merlin_data_preprocessor import ItalianMerlinDataPreprocessor
from src.data.preprocessors.portuguese_cople2_preprocessor import PortugueseCople2DataPreprocessor
from src.data.preprocessors.spanish_cedel2_data_preprocessor import SpanishCedel2DataPreprocessor
from src.utils import fix_seed


def main(args: argparse.Namespace):
    """Prepare data and save them in tsv format.

    :param args: (argparse.NameSpace) Arguments.

    Read examples of each dataset and save them
    in a tsv file with a name [lang]_[dataset] in data/raw.
    """

    fix_seed(seed=args.seed)

    dp_raw = os.path.join("data", "raw")
    if not os.path.exists(dp_raw):
        os.mkdir(dp_raw)

    for data_preprocessor in (
            CzechMerlinDataPreprocessor,
            EnglishWriteAndImproveDataPreprocessor,
            GermanMerlinDataPreprocessor,
            ItalianMerlinDataPreprocessor,
            PortugueseCople2DataPreprocessor,
            SpanishCedel2DataPreprocessor,
    ):
        print(data_preprocessor.__name__)
        data_preprocessor = data_preprocessor()
        print(data_preprocessor)
        data_preprocessor.to_tsv()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    args = parser.parse_args()

    main(args)
