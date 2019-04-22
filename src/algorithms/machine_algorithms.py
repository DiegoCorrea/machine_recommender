import pandas as pd
from sklearn.metrics import accuracy_score

from src.algorithms.k_neighbors_classifier import KNeighborsClassifier
from src.globalVariable import GlobalVariable


class MachineAlgorithms:

    @staticmethod
    def __train_naive_bayes(x_train, y_train):
        """
        Função de treino do classificador Naive Bayes que retorna uma instancia treinada
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        pass

    @staticmethod
    def __train_knn(x_train, x_test, y_train, y_test, run):
        """
        Função de treino do classificador KNN que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        y_pred = clf.predict_multiprocess(x_test)
        accuracy = MachineAlgorithms.__evaluate(y_pred, y_test)
        # logging.info("KNN Accuracy: " + str(accuracy))
        return pd.DataFrame(data=[[run, 'KNN', 'accuracy', accuracy]],
                            columns=GlobalVariable.results_column_name)

    @staticmethod
    def __train_tree(x_train, y_train):
        """
        Função de treino do classificador Decision Tree que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        pass

    @staticmethod
    def __train_perceptron(x_train, y_train):
        """
        Função de treino do classificador Perceptron que retorna uma instancia treinada
        As constantes estão no arquivo de variaveis de sistema
        :param x_train: Dados de treinamento
        :param y_train: Classes dos dados de treinamento
        :return: Classificador treinado
        """
        pass

    @staticmethod
    def __evaluate(y_pred, y_test):
        """
        Função que valida o treinamento com as métricas MAE e accuracy
        :param y_pred: Entrada predita
        :param y_test: Entrada real
        :return: Um par: primeiro o valor do MAE, segundo o valor do accuracy
        """
        return accuracy_score(y_test, y_pred)

    @staticmethod
    def main(x_train, x_test, y_train, y_test, run):
        """
        Função principal que executa todos os algoritmos selecionados, retornando um dataframe com os resultados
        :param x_train: Dados de treinamento
        :param x_test: Classes dos dados de treinamento
        :param y_train: Dados de teste
        :param y_test: Classes dos dados de teste
        :param run: Rodada do sistema
        :param model: Nome do modelo de dados a ser processado
        :return: DataFrame com os resultados de cada algoritmo
        """
        # logging.info("*" * 50)
        # Uso dos dados no treinamento e teste do KNN, por fim avaliação dos resultados
        result_df = MachineAlgorithms.__train_knn(x_train, x_test, y_train, y_test, run)
        # logging.info("*" * 50)
        return result_df
