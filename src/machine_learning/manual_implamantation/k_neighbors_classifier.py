from multiprocessing import Pool
from statistics import mode

import math
import pandas as pd

from src.globalVariable import GlobalVariable


class KNeighborsClassifier:
    t_data = pd.DataFrame()
    t_label = pd.DataFrame()

    def __init__(self, n_neighbors=GlobalVariable.k_neighbors):
        self.__DATASET_TRAIN = pd.DataFrame()
        self.__LABELS_TRAIN = pd.DataFrame()
        self.__N_NEIGHBORS = n_neighbors

    def fit(self, dataset_train, label):
        self.__DATASET_TRAIN = dataset_train
        KNeighborsClassifier.t_data = dataset_train
        self.__LABELS_TRAIN = pd.DataFrame(index=dataset_train.index.values.tolist())
        self.__LABELS_TRAIN['label'] = label
        KNeighborsClassifier.t_label = self.__LABELS_TRAIN

    @staticmethod
    def euclidean_distance(x_test, x_train):
        return math.sqrt(sum([(float(a) - float(b)) ** 2 for a, b in zip(x_test, x_train)]))

    def __neigh(self, x_test):
        x_train_array = self.__DATASET_TRAIN.values.tolist()
        data_pass = [(xt, x_test) for xt in x_train_array]
        pool = Pool(GlobalVariable.processor_number)
        result = pool.starmap(
            KNeighborsClassifier.euclidean_distance,
            data_pass
        )
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
    def neighbor_multi(x_test):
        index_list = KNeighborsClassifier.t_data.index.values
        distance = [KNeighborsClassifier.euclidean_distance(x_test=x_test, x_train=data) for data in
                    KNeighborsClassifier.t_data.values.tolist()]
        list__ = {a: b for a, b in zip(index_list, distance)}
        ordered_neigh = [(k, list__[k]) for k in sorted(list__, key=list__.get, reverse=False)]
        label_test = [KNeighborsClassifier.t_label.loc[i]['label'] for i, v in
                      ordered_neigh[:GlobalVariable.k_neighbors]]
        return mode(label_test)

    @staticmethod
    def predict_multiprocess(y_test_dataset):
        data_entry = [data.tolist() for index, data in y_test_dataset.iterrows()]
        # data_entry = [(index_list, test) for test in y_test_dataset]
        pool = Pool(GlobalVariable.processor_number)
        result = pool.map(
            KNeighborsClassifier.neighbor_multi,
            data_entry
        )
        pool.close()
        pool.join()
        return result

    @staticmethod
    def test():
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        import pandas as pd
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        knn = KNeighborsClassifier()
        knn.fit(dataset_train=df[3:149], label=data['target'][3:149])
        predict = knn.predict_multiprocess(df[0:2].tolist())
        print(predict)
        print(accuracy_score(data['target'][0:2], predict))
