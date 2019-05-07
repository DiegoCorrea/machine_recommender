from multiprocessing import Pool

import math
import pandas as pd

from src.globalVariable import GlobalVariable


class NaiveBayesClassifier:
    def __init__(self):
        self.__DATASET_TRAIN = pd.DataFrame()
        self.__LABELS_TRAIN = pd.DataFrame()
        self.__POS_CLASS = pd.DataFrame()
        self.__NEG_CLASS = pd.DataFrame()

    def fit(self, x_train_dataset, x_label_dataset):
        self.__DATASET_TRAIN = x_train_dataset
        self.__LABELS_TRAIN = pd.DataFrame(index=x_train_dataset.index.values.tolist())
        self.__LABELS_TRAIN['label'] = x_label_dataset
        self.__POS_CLASS = self.__LABELS_TRAIN[self.__LABELS_TRAIN['label'] == True]
        self.__NEG_CLASS = self.__LABELS_TRAIN[self.__LABELS_TRAIN['label'] == False]

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
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        ckf = NaiveBayesClassifier()
        ckf.fit(x_train_dataset=df[3:149], x_label_dataset=data['target'][3:149])
        predict = ckf.predict_multiprocess(df[0:2])
        print(predict)
        print(accuracy_score(data['target'][0:2], predict))

    @staticmethod
    def mean(numbers):
        return sum(numbers) / float(len(numbers))

    @staticmethod
    def stdev(numbers):
        avg = NaiveBayesClassifier.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)
