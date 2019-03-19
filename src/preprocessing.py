import re
import unicodedata
from multiprocessing.dummy import Pool as ThreadPool
import contractions
import inflect
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from pandas import concat


cachedStopWords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def strip_html(text):
    """
    Retira as TAGs HTML do texto
    :param text: texto a ser processado
    :return text: texto sem as tags html
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    """
    Remove texto entre cochetes
    :param text: texto a ser processado
    :return text: texto sem as frases entre cochetes
    """
    return re.sub('\[[^]]*\]', '', text)


def replace_contractions(text):
    """
    Troca as palavras contraidas para palavras independentes
    :param text: texto a ser processado
    :return text: texto sem palavras contraidas
    """
    return contractions.fix(text)


def remove_non_ascii(words):
    """
    Remove non-ASCII characters from list of tokenized words
    :param words:
    :return:
    """
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """
    Convert all characters to lowercase from list of tokenized words
    :param words:
    :return:
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """
    Remove punctuation from list of tokenized words
    :param words:
    :return:
    """
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """
    Replace all interger occurrences in list of tokenized words with textual representation
    :param words:
    :return:
    """
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """
    Remove stop words from list of tokenized words
    :param words:
    :return:
    """
    new_words = []
    for word in words:
        if word not in cachedStopWords:
            new_words.append(word)
    return new_words


def stem_words(words):
    """
    Stem words in list of tokenized words
    :param words:
    :return:
    """
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """
    Lemmatize verbs in list of tokenized words
    :param words:
    :return:
    """
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


def stem_and_lemmatize(words):
    stems = stem_words(words)
    # lemmas = lemmatize_verbs(words)
    return stems


def preprocessing_apply(split_dataset_df):
    # split_dataset_df['lemma_sentence'] = ""
    split_dataset_df['stem_sentence'] = ""
    for index, row in split_dataset_df.iterrows():
        sample = strip_html(row['sentence'])
        # sample = remove_between_square_brackets(sample)
        sample = replace_contractions(sample)
        bag_words = nltk.word_tokenize(sample)
        words = normalize(bag_words)
        stems = stem_and_lemmatize(words)
        split_dataset_df.at[index, 'stem_sentence'] = " ".join(str(x) for x in stems)
        # split_dataset_df.at[index, 'lemma_sentence'] = lemmas
    return split_dataset_df


def main_start(dataset_df):
    pool = ThreadPool(3)
    result = pool.map(preprocessing_apply, np.array_split(dataset_df, 3))
    pool.close()
    pool.join()
    return concat(result, sort=False)

