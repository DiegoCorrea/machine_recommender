import math
from multiprocessing.dummy import Pool as ThreadPool
import functools
from src.globalVariable import GlobalVariable


class KNN:
    def __init__(self, dataset, label, n_neighbors=3):
        self.__X_TRAIN = dataset
        self.__Y_TRAIN = label
        self.__n_neighbors = n_neighbors

    @staticmethod
    def __euclidean_distance(x_test, x_train):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_test, x_train)]))

    def __neigh(self, x_test):
        pool = ThreadPool(GlobalVariable.processor_number)
        result = pool.map(functools.partial(KNN.__euclidean_distance, tuple(x_test.values.tolist())),
                          self.__X_TRAIN.values.tolist())
        pool.close()
        pool.join()
        return result

    def predict(self, x_test):
        print(x_test.values.tolist())
        index_list = self.__X_TRAIN.index.values
        # for index, row in x_test.iterrows():
        test_neigh = self.__neigh(x_test)
        list__ = {a: b for a, b in zip(index_list, test_neigh)}
        s = [(k, list__[k]) for k in sorted(list__, key=list__.get, reverse=False)]
        print(s[:3])

    @staticmethod
    def test():
        from sklearn.datasets import load_iris
        import pandas as pd
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        knn = KNN(dataset=df.iloc[1:149], label=data['target'])
        knn.predict(df.iloc[0])
        return knn
