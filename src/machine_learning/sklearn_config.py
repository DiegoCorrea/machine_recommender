import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.evaluations.MAP.map_controller import MAPController
from src.globalVariable import GlobalVariable
from src.kemures.recommenders.UserAverage.user_average_controller import UserAverageController


class MachineAlgorithms:
    @staticmethod
    def train_naive_bayes(x_train, x_test, y_train, y_test, run):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = GaussianNB()
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        map_value = MAPController.get_ap_from_list(user_results_df['original_like'].tolist())
        accuracy, precision = MachineAlgorithms.evaluate(y_pred, y_test)
        output = pd.concat([pd.DataFrame(data=[[run, 'GaussianNB', 'accuracy', accuracy]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'GaussianNB', 'precision', precision]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'GaussianNB->COS', 'map', map_value]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'GaussianNB->COS', 'precision_cos',
                                                precision_score(positive_pred['original_like'],
                                                                positive_pred['pred_like'], average='micro')]],
                                         columns=GlobalVariable.results_column_name)
                            ])
        return output

    @staticmethod
    def train_knn(x_train, x_test, y_train, y_test, run):
        """
        Função de treino do classificador KNN que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = KNeighborsClassifier(n_neighbors=GlobalVariable.k_neighbors, n_jobs=GlobalVariable.processor_number)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        map_value = MAPController.get_ap_from_list(user_results_df['original_like'].tolist())
        accuracy, precision = MachineAlgorithms.evaluate(y_pred, y_test)
        output = pd.concat([pd.DataFrame(data=[[run, 'KNN', 'accuracy', accuracy]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'KNN', 'precision', precision]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'KNN->COS', 'map', map_value]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'KNN->COS', 'precision_cos',
                                                precision_score(positive_pred['original_like'],
                                                                positive_pred['pred_like'], average='micro')]],
                                         columns=GlobalVariable.results_column_name)
                            ])
        return output

    @staticmethod
    def train_tree(x_train, y_train):
        """
        Função de treino do classificador Decision Tree que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def train_perceptron(x_train, x_test, y_train, y_test, run):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = Perceptron(n_jobs=GlobalVariable.processor_number)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        map_value = MAPController.get_ap_from_list(user_results_df['original_like'].tolist())
        accuracy, precision = MachineAlgorithms.evaluate(y_pred, y_test)
        output = pd.concat([pd.DataFrame(data=[[run, 'Perceptron', 'accuracy', accuracy]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'Perceptron', 'precision', precision]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'Perceptron->COS', 'map', map_value]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'Perceptron->COS', 'precision_cos',
                                                precision_score(positive_pred['original_like'],
                                                                positive_pred['pred_like'], average='micro')]],
                                         columns=GlobalVariable.results_column_name)
                            ])
        return output

    @staticmethod
    def train_mlp(x_train, x_test, y_train, y_test, run):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = Perceptron(n_jobs=GlobalVariable.processor_number)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        map_value = MAPController.get_ap_from_list(user_results_df['original_like'].tolist())
        accuracy, precision = MachineAlgorithms.evaluate(y_pred, y_test)
        output = pd.concat([pd.DataFrame(data=[[run, 'MLP', 'accuracy', accuracy]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'MLP', 'precision', precision]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'MLP->COS', 'map', map_value]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, 'MLP->COS', 'precision_cos',
                                                precision_score(positive_pred['original_like'],
                                                                positive_pred['pred_like'], average='micro')]],
                                         columns=GlobalVariable.results_column_name)
                            ])
        return output

    @staticmethod
    def user_average(x_train_data, x_test_data, y_test_label, run):
        test_df = pd.DataFrame(data=[], index=x_test_data.index.values.tolist())
        test_df['like'] = y_test_label
        user_set = pd.concat([x_train_data, x_test_data])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train_data.index.values.tolist(),
                                                        song_model_ids=x_test_data.index.values.tolist())
        user_results_df = pd.concat([user_list, test_df], axis=1, sort=False, join='inner')
        return pd.DataFrame(
            data=[[run, 'COS', 'map', MAPController.get_ap_from_list(user_results_df['like'].tolist())]],
            columns=GlobalVariable.results_column_name)

    @staticmethod
    def evaluate(y_pred, y_test):
        """
        Função que valida o treinamento com as métricas MAE e accuracy
        :param y_pred: Entrada predita
        :param y_test: Entrada real
        :return: Um par: primeiro o valor do MAE, segundo o valor do accuracy
        """
        return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro')

    @staticmethod
    def main(x_train, x_test, y_train, y_test, run):
        result_df = MachineAlgorithms.train_knn(x_train, x_test, y_train, y_test, run)
        # result_df = pd.concat([result_df, MachineAlgorithms.train_perceptron(x_train, x_test, y_train, y_test, run)])
        result_df = pd.concat([result_df, MachineAlgorithms.train_naive_bayes(x_train, x_test, y_train, y_test, run)])
        return pd.concat([result_df, MachineAlgorithms.user_average(x_train_data=x_train, x_test_data=x_test,
                                                                    y_test_label=y_test, run=run)])
