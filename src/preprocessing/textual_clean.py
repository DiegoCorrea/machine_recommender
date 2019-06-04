import logging
import re
from multiprocessing import Pool

import contractions
import inflect
import nltk
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from src.globalVariable import GlobalVariable


class TextualClean:
    cachedStopWords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def strip_html(text):
        """
        Retira as TAGs HTML do texto
        :param text: texto a ser processado
        :return text: texto sem as tags html
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def remove_between_square_brackets(text):
        """
        Remove texto entre cochetes
        :param text: texto a ser processado
        :return text: texto sem as frases entre cochetes
        """
        return re.sub('\[[^]]*\]', '', text)

    @staticmethod
    def replace_contractions(text):
        """
        Troca as palavras contraidas para palavras independentes
        :param text: texto a ser processado
        :return text: texto sem palavras contraidas
        """
        return contractions.fix(text)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def remove_stopwords(words):
        """
        Remove stop words from list of tokenized words
        :param words:
        :return:
        """
        new_words = []
        for word in words:
            if word not in TextualClean.cachedStopWords:
                new_words.append(word)
        return new_words

    @staticmethod
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

    @staticmethod
    def lemmatize_verbs(words):
        """
        Lemmatize verbs in list of tokenized words
        :param words:
        :return:
        """
        lemmas = []
        for word in words:
            lemma = TextualClean.lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    @staticmethod
    def normalize(words):
        words = TextualClean.remove_non_ascii(words)
        words = TextualClean.to_lowercase(words)
        words = TextualClean.remove_punctuation(words)
        words = TextualClean.replace_numbers(words)
        words = TextualClean.remove_stopwords(words)
        return words

    @staticmethod
    def stem_and_lemmatize(words):
        stems = TextualClean.stem_words(words)
        # lemmas = TextualClean.lemmatize_verbs(words)
        return stems

    @staticmethod
    def preprocessing_apply(song_df):
        logging.info("Aplicando Limpeza 1")
        sample = TextualClean.strip_html(song_df['data'])
        logging.info("Aplicando Limpeza 2")
        # sample = remove_between_square_brackets(sample)
        sample = TextualClean.replace_contractions(sample)
        logging.info("Aplicando Limpeza 3")
        bag_words = nltk.word_tokenize(sample)
        logging.info("Aplicando Limpeza 4")
        words = TextualClean.normalize(bag_words)
        logging.info("Aplicando Limpeza 5")
        stems = TextualClean.stem_and_lemmatize(words)
        song_df['stem_data'] = " ".join(str(x) for x in stems)
        # split_dataset_df.at[index, 'lemma_sentence'] = lemmas
        return song_df

    @staticmethod
    def main_start(dataset_df):
        df = TextualClean.concat_fields(dataset_df)
        logging.info("Finalizando unificação")
        pool = Pool(GlobalVariable.processor_number)
        result = pool.map(TextualClean.preprocessing_apply,
                          df)
        pool.close()
        pool.join()
        logging.info("Concatenando resultados!")
        return pd.concat(result, sort=False)

    @staticmethod
    def concat_fields(dataset_df):
        dataset_df['stem_data'] = " "
        dataset_df['data'] = dataset_df['title'] + ' ' + dataset_df['album'] \
                                        + ' ' + dataset_df['artist'] + ' ' + dataset_df['gender']  \
                                        + ' ' + dataset_df['year']
        dataset_df.drop(['title', 'album', 'artist', 'year', 'gender'], inplace=True, axis=1)
        logging.info("Campos das músicas unificados!")
        return dataset_df
