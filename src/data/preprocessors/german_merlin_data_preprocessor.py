from src.data.preprocessors.merlin_data_preprocessor import MerlinDataPreprocessor


class GermanMerlinDataPreprocessor(MerlinDataPreprocessor):

    def __init__(self):
        super(GermanMerlinDataPreprocessor, self).__init__(
            lang="de",
            dataset="merlin",
        )


if __name__ == '__main__':
    GermanMerlinDataPreprocessor().to_tsv()
