from multiprocessing import Pool

import pandas as pd

from src.globalVariable import GlobalVariable


class PerceptronClassifier:
    t_data = pd.DataFrame()

    def __init__(self):
        self.__TRAIN_DATA = pd.DataFrame()
        self.__TRAIN_LABELS = pd.DataFrame()

    def fit(self, train_data, label_data):
        self.__TRAIN_DATA = train_data
        self.__TRAIN_LABELS = pd.DataFrame(index=train_data.index.values.tolist())
        self.__TRAIN_LABELS['label'] = label_data

    @staticmethod
    def __start():
        pass

    def predict(self, test_data):
        return None

    def predict_multiprocess(self, test_data):
        pool = Pool(GlobalVariable.processor_number)
        result = pool.starmap(
            PerceptronClassifier.__start,
            [(self.__TRAIN_DATA, self.__TRAIN_LABELS, test) for test in test_data]
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
        clf = PerceptronClassifier()
        clf.fit(train_data=df[3:149], label_data=data['target'][3:149])
        predict = clf.predict_multiprocess(df[0:2])
        print(predict)
        print(accuracy_score(data['target'][0:2], predict))
