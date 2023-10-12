from NewsGroupDataset import NewsGroupDataset
from SplitDataset import SplitDataset, DatasetSplitter
from LogisticRegression import LogisticRegression


dataset = NewsGroupDataset()
print(dataset.output.shape)
print(dataset.input.shape)
DatasetSplitter().split(dataset)

logistic_regression = LogisticRegression()
logistic_regression.fit()
