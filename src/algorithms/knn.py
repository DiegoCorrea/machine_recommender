import math
from multiprocessing.dummy import Pool as ThreadPool
import functools
from statistics import mode
from src.globalVariable import GlobalVariable


class KNN:
    def __init__(self, dataset, label, n_neighbors=3):
        self.__DATASET_TRAIN = dataset
        self.__LABELS_TRAIN = label
        self.__N_NEIGHBORS = n_neighbors

    @staticmethod
    def __euclidean_distance(x_test, x_train):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_test, x_train)]))

    def __neigh(self, x_test):
        pool = ThreadPool(GlobalVariable.processor_number)
        result = pool.map(functools.partial(KNN.__euclidean_distance, tuple(x_test.values.tolist())),
                          self.__DATASET_TRAIN.values.tolist())
        pool.close()
        pool.join()
        return result

    def predict_k(self, dataset_test):
        index_list = self.__DATASET_TRAIN.index.values
        all_neigh = []
        for index, row in dataset_test.iterrows():
            test_neigh = self.__neigh(row)
            list__ = {a: b for a, b in zip(index_list, test_neigh)}
            ordered_neigh = [(k, list__[k]) for k in sorted(list__, key=list__.get, reverse=False)]
            all_neigh.append({index: ordered_neigh[:self.__N_NEIGHBORS]})
        return all_neigh

    def predict(self, dataset_test):
        index_list = self.__DATASET_TRAIN.index.values
        all_neigh = []
        for index, row in dataset_test.iterrows():
            test_neigh = self.__neigh(row)
            list__ = {a: b for a, b in zip(index_list, test_neigh)}
            ordered_neigh = [(k, list__[k]) for k in sorted(list__, key=list__.get, reverse=False)]
            label_test = [self.__LABELS_TRAIN[i] for i, v in ordered_neigh[:self.__N_NEIGHBORS]]
            all_neigh.append({index: mode(label_test)})
        return all_neigh

    @staticmethod
    def test():
        from sklearn.datasets import load_iris
        import pandas as pd
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        knn = KNN(dataset=df[3:149], label=data['target'])
        predict = knn.predict(df[0:2])
        print(predict)
