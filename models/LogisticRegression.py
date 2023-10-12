from sklearn.metrics import accuracy_score, classification_report
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter
from sklearn.linear_model import LogisticRegression


dataset = NewsGroupDataset()
training_input, training_output, test_input, test_output = DatasetSplitter().split(dataset).as_list()

logistic_regression = LogisticRegression(random_state=42)

logistic_regression.fit(training_input, training_output)

predictions = logistic_regression.predict(test_input)
print("Accuracy:", accuracy_score(test_output, predictions))
print(classification_report(test_output, predictions))
