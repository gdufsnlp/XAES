import torch


class DataSplitForPLMs(torch.utils.data.Dataset):
    """Data split for pre-trained language models.

    Parameters:
        split: (str) train/dev/test.
        xs: Essays.
        ys: Scores.

    """

    def __init__(self, split: str, xs, ys, kwargs):
        self.split = split
        self.xs = xs
        self.ys = ys

        self.kwargs = kwargs
        self.qualities = self.add_writing_qualities()

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.xs.items()
        }
        item['labels'] = torch.tensor(self.ys[idx]).float()

        if self.kwargs["include_writing_qualities"]:
            if self.split == "train":
                item["qualities"] = torch.tensor(self.qualities[idx])
            elif self.split in ["dev", "test"]:
                item["qualities"] = None
            else:
                raise NotImplementedError

        return item

    def __len__(self):
        return len(self.ys)

    def add_writing_qualities(self):
        if not self.kwargs["include_writing_qualities"]:
            return []

        qualities = []
        for y in self.ys:
            y = round(y, 1)

            if self.kwargs["writing_quality_granularity"] == 2:
                if y <= 0.5:
                    # 0, 0.2, 0.4.
                    qualities.append(0)
                else:
                    # 0.6, 0.8, 1.0.
                    qualities.append(1)
            elif self.kwargs["writing_quality_granularity"] == 3:
                if y <= 0.2:
                    # Low quality.
                    # For cefr: 0, 0.2.
                    qualities.append(0)
                elif 0.2 < y <= 0.7:
                    # Medium quality.
                    # For cefr: 0.4, 0.6.
                    qualities.append(1)
                elif y > 0.7:
                    # High quality.
                    # For cefr: 0.8, 1.0.
                    qualities.append(2)
            elif self.kwargs["writing_quality_granularity"] == 6:
                qualities.append(
                    {
                        0.0: 0,
                        0.2: 1,
                        0.4: 2,
                        0.6: 3,
                        0.8: 4,
                        1.0: 5,
                    }[y]
                )

        return qualities