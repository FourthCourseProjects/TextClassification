from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


class NewsGroupDataset:
    def __init__(self):
        self.input = self.vectorize(fetch_20newsgroups().get("data")).toarray()
        self.output = self.one_hot_encode(fetch_20newsgroups().get("target"))

    def vectorize(self, documents):
        return CountVectorizer(stop_words='english', max_features=10000).fit_transform(documents)

    def one_hot_encode(self, targets):
        return OneHotEncoder().fit_transform(targets.reshape(-1, 1))

    def __len__(self):
        return len(self.input)
