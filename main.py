from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter

dataset = NewsGroupDataset()
print(dataset.output.shape)
print(dataset.input.shape)
DatasetSplitter().split(dataset)
