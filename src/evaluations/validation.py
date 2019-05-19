from sklearn.model_selection import train_test_split, StratifiedKFold

from src.globalVariable import GlobalVariable


class Validation:
    @staticmethod
    def split_data(data, labels):
        """
        Função particionadora dos dados em treino e teste
        As constantes estão no arquivo de variaveis de sistema
        :param data: Dados atributo/valor
        :param labels: Classe dos dados
        :return: Quatro valores: dados de treino, classe de treino, dados de teste, classe de teste
        """
        return train_test_split(data, labels, test_size=GlobalVariable.test_set_size, stratify=labels)

    @staticmethod
    def split_by_index(index_list, polarity_class):
        return train_test_split(index_list, polarity_class, test_size=GlobalVariable.test_set_size)

    @staticmethod
    def split_tfidf(tfidf_matrix, x_train, x_test):
        return tfidf_matrix.ix[x_train], tfidf_matrix.ix[x_test]

    @staticmethod
    def split_pmi(pmi_model, x_train, x_test):
        return pmi_model.ix[x_train], pmi_model.ix[x_test]

    @staticmethod
    def stratified_split_data(data, labels):
        skf = StratifiedKFold(n_splits=5, random_state=0)
        X_train, X_test = None, None
        y_train, y_test = None, None
        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        return X_train, X_test, y_train, y_test
