import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class NewsGroupDataset:
    def __init__(self):
        self.input = self.vectorize(fetch_20newsgroups().get("data"))
        self.output = self.label_encode(fetch_20newsgroups().get("target"))

    def vectorize(self, documents):
        return CountVectorizer(stop_words='english', max_features=10000).fit_transform(documents)

    def label_encode(self, targets):
        return LabelEncoder().fit_transform(targets.reshape(-1, 1))

    def __len__(self):
        return len(self.input)
