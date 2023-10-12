from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter

dataset = NewsGroupDataset()
svm = svm.SVR()
split_dataset = DatasetSplitter().split(dataset)

svm.fit(split_dataset.training_input, split_dataset.training_output)
predictions = svm.predict(split_dataset.test_input)

print("Accuracy:", accuracy_score(split_dataset.test_output, predictions))
print(classification_report(split_dataset.test_output, predictions))
