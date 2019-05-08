import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class FrequencyModel:
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
        word_position = count_vec.vocabulary_.items()
        # Calcula o IDF a partir do TF
        tfidf_tran = TfidfTransformer(norm="l2")
        tfidf_tran.fit(tf_matrix)
        tfidf_matrix = tfidf_tran.transform(tf_matrix)
        # print(tfidf_matrix.todense())
        return np.array(tfidf_matrix.toarray()), word_position

    @staticmethod
    def mold(original_dataset):
        """
        Função que transforma a entrada original em um modelo de TF-IDF
        :param original_dataset: DataFrame com as entradas
        :return:
        """
        # print(original_dataset.head())
        logging.info("Gerando matrix do TF-IDF...")
        tfidf_matrix, word_position = FrequencyModel.tf_as_matrix(sentence_list=original_dataset['stem_data'].tolist())
        logging.info("Adicionando nome das colunas e as index na matrix TF-IDF")
        index_list = original_dataset.song_id.tolist()
        columns_label = [a for a, v in sorted(word_position, key=lambda k: (k[1], k[0]))]
        print(tfidf_matrix)
        print(np.intp)
        # data_entry = np.matrix(tfidf_matrix)
        return pd.DataFrame(data=tfidf_matrix, columns=columns_label, index=index_list)
