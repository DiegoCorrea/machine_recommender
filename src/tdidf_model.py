import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
    tf_matrix = count_vec.transform(sentence_list).toarray()
    # Retorna os nomes das colunas que são palavras
    word_position = count_vec.vocabulary_.items()
    # Calcula o IDF a partir do TF
    tfidf_tran = TfidfTransformer(norm="l2")
    tfidf_tran.fit(tf_matrix)
    tfidf_matrix = tfidf_tran.transform(tf_matrix)
    return tfidf_matrix.toarray(), sorted(word_position, key=lambda k: (k[1], k[0]))


def mold(original_dataset):
    """
    Função que transforma a entrada original em um modelo de TF-IDF
    :param original_dataset: DataFrame com as entradas
    :return:
    """
    tfidf_matrix, word_position = tf_as_matrix(sentence_list=original_dataset['stem_sentence'].tolist())
    tfidf_pattern = pd.DataFrame(data=np.matrix(tfidf_matrix), columns=[a for a, v in word_position])
    return tfidf_pattern
