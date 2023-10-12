from sklearn.metrics import accuracy_score, classification_report
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter
from sklearn.linear_model import LogisticRegression

# predictions = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
# print(classification_report(y_test, predictions))

dataset = NewsGroupDataset()
# print(dataset.output.shape)
# print(dataset.input.shape)
training_input, training_output, test_input, test_output = DatasetSplitter().split(dataset).as_list()

logistic_regression = LogisticRegression(random_state=42)

logistic_regression.fit(training_input, training_output)

predictions = logistic_regression.predict(test_input)
print("Accuracy:", accuracy_score(test_output, predictions))
print(classification_report(test_output, predictions))
