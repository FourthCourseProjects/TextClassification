from sklearn.model_selection import train_test_split


class DatasetSplitter:
    def split(self, dataset, proportion=0.8):
        split_dataset = SplitDataset(
            train_test_split(dataset.input, dataset.output, test_size=1 - proportion, random_state=42))
        return split_dataset


class SplitDataset:
    def __init__(self, tuple):
        self.training_input = tuple[0]
        self.test_input = tuple[1]
        self.training_output = tuple[2]
        self.test_output = tuple[3]

    def as_list(self):
        return [self.training_input, self.training_output, self.test_input, self.test_output]
