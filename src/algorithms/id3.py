from src.config.machine import Machine
from math import log2


class ID3:
    def __init__(self, dataset, label):
        self.DATASET = dataset
        self.LABEL = label

    @staticmethod
    def entropy(class_a, class_b, total):
        return -(class_a/total)*log2(class_a/total) - (class_b/total)*log2(class_b/total)

    def gain(self, column_name):
        names_uniques_attrs = self.DATASET[column_name].unique().tolist()
        count_row = self.DATASET.shape[0]
        values_count = self.DATASET[column_name].value_counts()
        print("Value Count")
        print(values_count)
        for attr in names_uniques_attrs:
            pass

    @staticmethod
    def test():
        from sklearn.datasets import load_iris
        import pandas as pd
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        id3 = ID3(dataset=df, label=data['target'])
        return id3