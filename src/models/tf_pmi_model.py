import math
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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
    new_column_name = [a for a, b in sorted(word_position, key=lambda k: (k[1], k[0]))]
    tf_matrix_df = pd.DataFrame(data=tf_matrix, columns=new_column_name)
    return tf_matrix_df


def __calc_pmi(column):
    global words_count
    global negative_filter_df
    global positive_filter_df
    global total_words
    global piw
    words_count[column] = dict()
    words_count[column]['negative'] = negative_filter_df[column].sum()
    words_count[column]['positive'] = positive_filter_df[column].sum()
    total_w = words_count[column]['positive'] + words_count[column]['negative']
    words_count[column]['negative'] = words_count[column]['negative'] / total_words
    words_count[column]['positive'] = words_count[column]['positive'] / total_words
    words_count[column]['total'] = total_w / total_words
    result = dict()
    result['positive'] = words_count[column]['positive'] / (words_count[column]['total'] * piw[1])
    result['negative'] = words_count[column]['negative'] / (words_count[column]['total'] * piw[1])
    if result['negative'] == 0.0:
        result['negative'] = 0.00001
    if result['positive'] == 0.0:
        result['positive'] = 0.00001
    return (column,
            math.log2(result['positive']),
            math.log2(result['negative'])
            )


words_count = dict()
negative_filter_df = None
positive_filter_df = None
total_words = 0
piw = []


def pmi(tf_matrix_df):
    global words_count
    global negative_filter_df
    global positive_filter_df
    global total_words
    global piw
    negative_filter_df = deepcopy(tf_matrix_df[tf_matrix_df['__POLARITY__'] == 0])
    positive_filter_df = deepcopy(tf_matrix_df[tf_matrix_df['__POLARITY__'] == 1])
    negative_filter_df.drop(['__POLARITY__'], axis=1, inplace=True)
    positive_filter_df.drop(['__POLARITY__'], axis=1, inplace=True)
    tf_df = deepcopy(tf_matrix_df.drop(['__POLARITY__'], axis=1))
    piw = pd.value_counts(tf_matrix_df['__POLARITY__'].values, sort=False)
    piw = piw / (piw[0] + piw[1])
    total_words = sum([tf_df[col].sum() for col in tf_df.columns])
    pool = ThreadPool(3)
    map_results = pool.map(__calc_pmi, tf_df.columns)
    pool.close()
    pool.join()
    for column, positive, negative in map_results:
        negative_filter_df[column] = negative_filter_df[column] * negative
        positive_filter_df[column] = positive_filter_df[column] * positive
    return pd.concat([positive_filter_df, negative_filter_df], sort=False)


def mold(original_dataset):
    """
    Função que transforma a entrada original em um modelo de TF-IDF
    :param original_dataset: DataFrame com as entradas
    :return:
    """
    tf_matrix_df = tf_as_matrix(sentence_list=original_dataset['stem_sentence'].tolist())
    tf_matrix_df['__POLARITY__'] = original_dataset['polarity']
    return pmi(tf_matrix_df)
