from src.data.preprocessors.merlin_data_preprocessor import MerlinDataPreprocessor


class ItalianMerlinDataPreprocessor(MerlinDataPreprocessor):

    def __init__(self):
        super(ItalianMerlinDataPreprocessor, self).__init__(
            lang="it",
            dataset="merlin",
        )


if __name__ == '__main__':
    ItalianMerlinDataPreprocessor().to_tsv()
