import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class FrequencyModel:

    @staticmethod
    def __count(count_vec, sentence_list):
        grow = 0
        jump = 1000
        sky = jump
        tf_matrix_list = []
        while grow < len(sentence_list):
            # tf_matrix_list.append(count_vec.transform(sentence_list[grow:sky]).toarray())
            tf_matrix_list = np.concatenate((tf_matrix_list, count_vec.transform(sentence_list[grow:sky]).toarray()))
            grow = sky
            sky += jump
        return np.concatenate(tf_matrix_list)

    @staticmethod
    def tf_idf_tran(tfidf_tran, tf_matrix):
        grow = 0
        jump = 1000
        sky = jump
        tf_matrix_list = []
        while grow < len(tf_matrix):
            tf_matrix_list.append(tfidf_tran.transform(tf_matrix[grow:sky]))
            grow = sky
            sky += jump
        return tf_matrix_list

    @staticmethod
    def tf_as_matrix(sentence_list):
        """
        Transforma a lista de sentenças recebidas em uma matriz onde a coluna são as palavras, as tinhas as sentenças
        E os valores o TF-IDF
        :param sentence_list: Vetor de Vetores com as sentenças em unigram
        :return: Matriz com os valores do TF-IDF
        """
        # Realiza uma contagem os itens nos vetores
        count_vec = CountVectorizer()
        count_vec.fit_transform(sentence_list)
        tf_matrix = count_vec.transform(sentence_list)
        # Retorna os nomes das colunas que são palavras
        # tf_matrix = FrequencyModel.__count(count_vec, sentence_list)
        print(tf_matrix)
        word_position = count_vec.vocabulary_.items()
        # Calcula o IDF a partir do TF
        tfidf_tran = TfidfTransformer(norm="l2")
        tfidf_tran.fit(tf_matrix)
        tfidf_matrix = tfidf_tran.transform(tf_matrix)
        # tfidf_matrix_list = FrequencyModel.tf_idf_tran(tfidf_tran, tf_matrix)
        # print(tfidf_matrix_list)
        # tfidf_matrix = np.concatenate(tfidf_matrix_list)
        return tfidf_matrix.toarray(), sorted(word_position, key=lambda k: (k[1], k[0]))

    @staticmethod
    def mold(original_dataset):
        """
        Função que transforma a entrada original em um modelo de TF-IDF
        :param original_dataset: DataFrame com as entradas
        :return:
        """
        print(original_dataset.head())
        tfidf_matrix, word_position = FrequencyModel.tf_as_matrix(sentence_list=original_dataset['stem_data'].tolist())
        tfidf_pattern = pd.DataFrame(data=np.matrix(tfidf_matrix), columns=[a for a, v in word_position])
        return tfidf_pattern
