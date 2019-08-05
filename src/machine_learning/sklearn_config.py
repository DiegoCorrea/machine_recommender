import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluations.map_metric import MAPController
from src.evaluations.mrr_metric import MRRController
from src.globalVariable import GlobalVariable
from src.kemures.recommenders.UserAverage.user_average_controller import UserAverageController


class MachineAlgorithms:
    @staticmethod
    def rank_evaluation(run, scenario, algorithm, relevance_array):
        output = pd.DataFrame()
        map_score_list = MAPController.at_all_position(relevance_array)
        mrr_score_list = MRRController.at_all_position(relevance_array)
        for at, map_score, mrr_score in zip(GlobalVariable.AT_SIZE_LIST, map_score_list, mrr_score_list):
            output = pd.concat([output,
                                pd.DataFrame(data=[[run, scenario, algorithm, 'map', at, map_score]],
                                             columns=GlobalVariable.results_column_name),
                                pd.DataFrame(data=[[run, scenario, algorithm, 'mrr', at, mrr_score]],
                                             columns=GlobalVariable.results_column_name)
                                ])
        return output

    @staticmethod
    def train_linear_regressor(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = LinearRegression()
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        map_value = MAPController.get_ap_from_list(user_results_df['original_like'].tolist())
        mrr_value = MRRController.get_rr_from_list(user_results_df['original_like'].tolist())
        output = pd.concat([pd.DataFrame(data=[[run, scenario, 'LR', 'map', map_value]],
                                         columns=GlobalVariable.results_column_name),
                            pd.DataFrame(data=[[run, scenario, 'LR', 'mrr', mrr_value]],
                                         columns=GlobalVariable.results_column_name)
                            ])
        return output

    @staticmethod
    def train_random_forest(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="RF",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_decision_tree(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="CART",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_svm_svc(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = SVC(gamma='scale', decision_function_shape='ovo')
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="SVC",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_naive_bayes(x_train, x_test, y_train, y_test, run, scenario):
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
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="GNB",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_knn(x_train, x_test, y_train, y_test, run, scenario):
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
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="KNN",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_perceptron(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = Perceptron(max_iter=1000, tol=1e-3, n_jobs=GlobalVariable.processor_number)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="PER",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def train_mlp(x_train, x_test, y_train, y_test, run, scenario):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = MLPClassifier(max_iter=1000, tol=1e-3, hidden_layer_sizes=5)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        test_results = pd.DataFrame(data=[], index=x_test.index.values.tolist())
        test_results['pred_like'] = y_pred
        test_results['original_like'] = y_test
        positive_pred = test_results[test_results['pred_like'] == True]
        candidate_songs = x_test.loc[positive_pred.index.values.tolist()]
        if len(candidate_songs) == 0:
            return pd.DataFrame()
            # logging.info("User has no candidate songs!")
        user_set = pd.concat([x_train, candidate_songs])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train.index.values.tolist(),
                                                        song_model_ids=candidate_songs.index.values.tolist())
        user_results_df = pd.concat([user_list, positive_pred], axis=1, sort=False)
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="MLP",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def user_average(x_train_data, x_test_data, y_test_label, run, scenario):
        test_df = pd.DataFrame(data=[], index=x_test_data.index.values.tolist())
        test_df['original_like'] = y_test_label
        user_set = pd.concat([x_train_data, x_test_data])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        user_list = UserAverageController.start_ranking(similarity_data_df=cos,
                                                        user_model_ids=x_train_data.index.values.tolist(),
                                                        song_model_ids=x_test_data.index.values.tolist())
        user_results_df = pd.concat([user_list, test_df], axis=1, sort=False, join='inner')
        return MachineAlgorithms.rank_evaluation(run=run, scenario=scenario, algorithm="recModel",
                                                 relevance_array=user_results_df['original_like'].values.tolist())

    @staticmethod
    def main(x_train, x_test, y_train, y_test, run, scenario):
        result_df = MachineAlgorithms.train_knn(x_train, x_test, y_train, y_test, run, scenario)
        result_df = pd.concat(
            [result_df, MachineAlgorithms.train_perceptron(x_train, x_test, y_train, y_test, run, scenario)])
        result_df = pd.concat([result_df, MachineAlgorithms.train_mlp(x_train, x_test, y_train, y_test, run, scenario)])
        result_df = pd.concat(
            [result_df, MachineAlgorithms.train_decision_tree(x_train, x_test, y_train, y_test, run, scenario)])
        result_df = pd.concat(
            [result_df, MachineAlgorithms.train_random_forest(x_train, x_test, y_train, y_test, run, scenario)])
        result_df = pd.concat(
            [result_df, MachineAlgorithms.train_naive_bayes(x_train, x_test, y_train, y_test, run, scenario)])
        result_df = pd.concat(
            [result_df, MachineAlgorithms.train_svm_svc(x_train, x_test, y_train, y_test, run, scenario)])
        return pd.concat([result_df, MachineAlgorithms.user_average(x_train_data=x_train, x_test_data=x_test,
                                                                    y_test_label=y_test, run=run, scenario=scenario)])
