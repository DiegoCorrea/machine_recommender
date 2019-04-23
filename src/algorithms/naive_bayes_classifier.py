from multiprocessing import Pool
from statistics import mode

import math
import pandas as pd

from src.globalVariable import GlobalVariable


class NaiveBayesClassifier:
    def __init__(self):
        self.__DATASET_TRAIN = pd.DataFrame()
        self.__LABELS_TRAIN = pd.DataFrame()

    def fit(self, dataset_train, label):
        self.__DATASET_TRAIN = dataset_train
        self.__LABELS_TRAIN = pd.DataFrame(index=dataset_train.index.values.tolist())
        self.__LABELS_TRAIN['label'] = label

    def predict(self, y_test_dataset):
        result = []
        for index, row in y_test_dataset:
            result = NaiveBayesClassifier.__prob_calc()
        return result

    @staticmethod
    def __prob_calc(x_train_dataset, x_label_dataset, x_test):
        return True

    def predict_multiprocess(self, y_test_dataset):
        pool = Pool(GlobalVariable.processor_number)
        result = pool.starmap(
            NaiveBayesClassifier.__prob_calc,
            [(self.__DATASET_TRAIN, self.__LABELS_TRAIN, test) for test in y_test_dataset]
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
        knn = NaiveBayesClassifier()
        knn.fit(dataset_train=df[3:149], label=data['target'][3:149])
        predict = knn.predict_multiprocess(df[0:2])
        print(predict)
        print(accuracy_score(data['target'][0:2], predict))
