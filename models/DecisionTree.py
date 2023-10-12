from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter

dataset = NewsGroupDataset()
training_input, training_output, test_input, test_output = DatasetSplitter().split(dataset).as_list()
tree = DecisionTreeClassifier(criterion='gini')

tree.fit(training_input, training_output)
predictions = tree.predict(test_input)
print("Accuracy:", accuracy_score(test_output, predictions))
print(classification_report(test_output, predictions))
