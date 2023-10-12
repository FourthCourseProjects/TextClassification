from sklearn.metrics import accuracy_score, classification_report
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter
from sklearn.cluster import KMeans


dataset = NewsGroupDataset()
training_input, training_output, test_input, test_output = DatasetSplitter().split(dataset).as_list()

kmeans = KMeans()

kmeans.fit(training_input, training_output)

predictions = kmeans.predict(test_input)
print("Accuracy:", accuracy_score(test_output, predictions))
print(classification_report(test_output, predictions))