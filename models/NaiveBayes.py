from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter


naive_bayes = MultinomialNB()
dataset = NewsGroupDataset()
split_dataset = DatasetSplitter().split(dataset)

naive_bayes.fit(split_dataset.training_input, split_dataset.training_output)
predictions = naive_bayes.predict(split_dataset.test_input)

print("Accuracy:", accuracy_score(split_dataset.test_output, predictions))
print(classification_report(split_dataset.test_output, predictions))
