import math
from multiprocessing.dummy import Pool as ThreadPool
import functools


class KNN:
    def __init__(self, x_train, y_train, n_neighbors=3, n_jobs=3):
        self.__X_TRAIN = x_train
        self.__Y_TRAIN = y_train
        self.__n_neighbors = n_neighbors
        self.__n_jobs = n_jobs

    @staticmethod
    def __euclidean_distance(x_test, x_train):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_test.values.tolist(), x_train.values.tolist())]))

    def __neigh(self, x_test):
        pool = ThreadPool(self.__n_jobs)
        result = pool.map(functools.partial(KNN.__euclidean_distance, dimension_x=x_test), self.__X_TRAIN)
        pool.close()
        pool.join()

    def predict(self, x_test):
        for index, row in x_test.iterrows():
            self.__neigh(row)

