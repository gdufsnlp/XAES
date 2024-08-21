from src.data.preprocessors.merlin_data_preprocessor import MerlinDataPreprocessor


class CzechMerlinDataPreprocessor(MerlinDataPreprocessor):

    def __init__(self):
        super(CzechMerlinDataPreprocessor, self).__init__(
            lang="cs",
            dataset="merlin",
        )


if __name__ == '__main__':
    CzechMerlinDataPreprocessor().to_tsv()
