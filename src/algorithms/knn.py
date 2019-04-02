import math
from multiprocessing.dummy import Pool as ThreadPool


class KNN:
    def __init__(self, dataset, label, k=3, n_jobs=3):
        self.DATASET = dataset
        self.label = label
        self.K = k

    def fit_transform(self):
        pass

    def __euclidian_distance(self, dimension_x, dimension_y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(dimension_x, dimension_y)]))

    def __neigh(self, x_test, y_test):
        pool = ThreadPool(self.K)
        result = pool.map(self.__euclidian_distance,
                            self.x_test)
        pool.close()
        pool.join()

    def predict(self, x_test, y_test):
        for index, row in x_test.iterrows():
            self.__neigh(x_test, row)

