import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.globalVariable import GlobalVariable
from src.kemures.recommenders.UserAverage.user_average_controller import UserAverageController


class MachineAlgorithms:
    @staticmethod
    def train_naive_bayes(x_train, y_train):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = GaussianNB()
        clf = clf.fit(x_train, y_train)
        return clf

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
        accuracy = MachineAlgorithms.evaluate(y_pred, y_test)
        ml_pred_result = pd.DataFrame(data=[[run, 'KNN', 'accuracy', accuracy]],
                                      columns=GlobalVariable.results_column_name)
        predction_to_users = pd.DataFrame(index=x_test.index.values.tolist())
        predction_to_users['pred_like'] = y_pred
        user_set = pd.concat([x_train, x_test])
        cos = pd.DataFrame(data=np.matrix(cosine_similarity(X=user_set)), columns=user_set.index.tolist(),
                           index=user_set.index.tolist())
        u = UserAverageController.start_ranking(similarity_data_df=cos,
                                                user_model_ids=x_train.index.values.tolist(),
                                                song_model_ids=x_test.index.values.tolist())
        print(u)
        return ml_pred_result

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
    def train_perceptron(x_train, y_train):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = Perceptron(n_jobs=GlobalVariable.processor_number)
        clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def evaluate(y_pred, y_test):
        """
        Função que valida o treinamento com as métricas MAE e accuracy
        :param y_pred: Entrada predita
        :param y_test: Entrada real
        :return: Um par: primeiro o valor do MAE, segundo o valor do accuracy
        """
        return accuracy_score(y_test, y_pred)

    # def main(x_train, x_test, y_train, y_test, run):
    #     """
    #     Função principal que executa todos os algoritmos selecionados, retornando um dataframe com os resultados
    #     :param x_train: Dados de treinamento
    #     :param x_test: Classes dos dados de treinamento
    #     :param y_train: Dados de teste
    #     :param y_test: Classes dos dados de teste
    #     :param run: Rodada do sistema
    #     :param model: Nome do modelo de dados a ser processado
    #     :return: DataFrame com os resultados de cada algoritmo
    #     """
    #     result_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    #     # Uso dos dados no treinamento e teste do Perceptron RNA, por fim avaliação dos resultados
    #     print("\t\tPerceptron")
    #     clf = train_perceptron(x_train, y_train)
    #     y_pred = clf.predict(x_test)
    #     mae, accuracy = evaluate(y_pred, y_test)
    #     result_df = pd.concat([result_df,
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'Perceptron', 'accuracy', accuracy]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'Perceptron', 'mae', mae]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    #                            ],
    #                           sort=False
    #                           )
    #     # Uso dos dados no treinamento e teste da Árvore de Decisão, por fim avaliação dos resultados
    #     print("\t\tÁrvore de Decisão")
    #     clf = train_tree(x_train, y_train)
    #     y_pred = clf.predict(x_test)
    #     mae, accuracy = evaluate(y_pred, y_test)
    #     result_df = pd.concat([result_df,
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'AD', 'accuracy', accuracy]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'AD', 'mae', mae]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    #                            ],
    #                           sort=False
    #                           )
    #     # Uso dos dados no treinamento e teste do KNN, por fim avaliação dos resultados
    #     print("\t\tKNN")
    #     clf = train_knn(x_train, y_train)
    #     y_pred = clf.predict(x_test)
    #     mae, accuracy = evaluate(y_pred, y_test)
    #     result_df = pd.concat([result_df,
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'KNN', 'accuracy', accuracy]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'KNN', 'mae', mae]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    #                            ],
    #                           sort=False
    #                           )
    #     # Uso dos dados no treinamento e teste do Naive Bayes, por fim avaliação dos resultados
    #     print("\t\tNaive Bayes")
    #     clf = train_naive_bayes(x_train, y_train)
    #     y_pred = clf.predict(x_test)
    #     mae, accuracy = evaluate(y_pred, y_test)
    #     result_df = pd.concat([result_df,
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'NB', 'accuracy', accuracy]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
    #                            pd.DataFrame(data=[[run, 'config_2', model, 'NB', 'mae', mae]],
    #                                         columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    #                            ],
    #                           sort=False
    #                           )
    #     return result_df

    @staticmethod
    def main(x_train, x_test, y_train, y_test, run):
        result_df = MachineAlgorithms.train_knn(x_train, x_test, y_train, y_test, run)
        return result_df
