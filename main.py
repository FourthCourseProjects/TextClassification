from NewsGroupDataset import NewsGroupDataset
from SplitDataset import SplitDataset, DatasetSplitter

dataset = NewsGroupDataset()
print(dataset.output.shape)
print(dataset.input.shape)
DatasetSplitter().split(dataset)
