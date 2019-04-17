import math
from multiprocessing.dummy import Pool as ThreadPool
import functools
from statistics import mode
from src.globalVariable import GlobalVariable
import pandas as pd


class KNeighborsClassifier:
    def __init__(self, n_neighbors=GlobalVariable.k_neighbors):
        self.__DATASET_TRAIN = pd.DataFrame()
        self.__LABELS_TRAIN = pd.DataFrame()
        self.__N_NEIGHBORS = n_neighbors

    def fit(self, dataset, label):
        self.__DATASET_TRAIN = dataset
        self.__LABELS_TRAIN = pd.DataFrame(index=dataset.index.values.tolist())
        self.__LABELS_TRAIN['label'] = label

    @staticmethod
    def __euclidean_distance(x_test, x_train):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_test, x_train)]))

    def __neigh(self, x_test):
        pool = ThreadPool(GlobalVariable.processor_number)
        result = pool.map(functools.partial(KNeighborsClassifier.__euclidean_distance, tuple(x_test.values.tolist())),
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
            label_test = [self.__LABELS_TRAIN.loc[i]['label'] for i, v in ordered_neigh[:self.__N_NEIGHBORS]]
            all_neigh.append(mode(label_test))
        return all_neigh

    @staticmethod
    def test():
        from sklearn.datasets import load_iris
        import pandas as pd
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        knn = KNeighborsClassifier()
        knn.fit(dataset=df[3:149], label=data['target'])
        predict = knn.predict(df[0:2])
        print(predict)
