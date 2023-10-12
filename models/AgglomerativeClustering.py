from sklearn.metrics import accuracy_score, classification_report
from NewsGroupDataset import NewsGroupDataset
from SplitDataset import DatasetSplitter
from sklearn.cluster import AgglomerativeClustering


dataset = NewsGroupDataset()
training_input, training_output, test_input, test_output = DatasetSplitter().split(dataset).as_list()

agg_cluster = AgglomerativeClustering()

predicted_labels = agg_cluster.fit_predict(training_input.toarray(), training_output)

print("Accuracy:", accuracy_score(training_output, predicted_labels))
print(classification_report(training_output, predicted_labels, zero_division=0))