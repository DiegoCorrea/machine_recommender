from multiprocessing import Pool

import pandas as pd

from src.globalVariable import GlobalVariable


class NaiveBayesClassifier:
    def __init__(self):
        self.__DATASET_TRAIN = pd.DataFrame()
        self.__LABELS_TRAIN = pd.DataFrame()

    def fit(self, x_train_dataset, x_label_dataset):
        self.__DATASET_TRAIN = x_train_dataset
        self.__LABELS_TRAIN = pd.DataFrame(index=x_train_dataset.index.values.tolist())
        self.__LABELS_TRAIN['label'] = x_label_dataset
        lt = self.__LABELS_TRAIN['label'].unique().tolist()

    def separateByClass(self):
        separated = {}
        for i in self.__LABELS_TRAIN['label'].unique().tolist():
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

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

    def predict_seq(self, y_test_dataset):
        for instance in y_test_dataset:
