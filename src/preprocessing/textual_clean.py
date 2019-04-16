import re
import unicodedata
from multiprocessing.dummy import Pool as ThreadPool
import contractions
import inflect
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd
from src.globalVariable import GlobalVariable


class TextualClean:
    cachedStopWords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def __strip_html(text):
        """
        Retira as TAGs HTML do texto
        :param text: texto a ser processado
        :return text: texto sem as tags html
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def __remove_between_square_brackets(text):
        """
        Remove texto entre cochetes
        :param text: texto a ser processado
        :return text: texto sem as frases entre cochetes
        """
        return re.sub('\[[^]]*\]', '', text)

    @staticmethod
    def __replace_contractions(text):
        """
        Troca as palavras contraidas para palavras independentes
        :param text: texto a ser processado
        :return text: texto sem palavras contraidas
        """
        return contractions.fix(text)

    @staticmethod
    def __remove_non_ascii(words):
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
    def __to_lowercase(words):
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
    def __remove_punctuation(words):
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
    def __replace_numbers(words):
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
    def __remove_stopwords(words):
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
    def __stem_words(words):
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
    def __lemmatize_verbs(words):
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
    def __normalize(words):
        words = TextualClean.__remove_non_ascii(words)
        words = TextualClean.__to_lowercase(words)
        words = TextualClean.__remove_punctuation(words)
        words = TextualClean.__replace_numbers(words)
        words = TextualClean.__remove_stopwords(words)
        return words

    @staticmethod
    def __stem_and_lemmatize(words):
        stems = TextualClean.__stem_words(words)
        # lemmas = TextualClean.__lemmatize_verbs(words)
        return stems

    @staticmethod
    def __preprocessing_apply(song_df):
        sample = TextualClean.__strip_html(song_df['data'])
        # sample = __remove_between_square_brackets(sample)
        sample = TextualClean.__replace_contractions(sample)
        bag_words = nltk.word_tokenize(sample)
        words = TextualClean.__normalize(bag_words)
        stems = TextualClean.__stem_and_lemmatize(words)
        song_df['stem_data'] = " ".join(str(x) for x in stems)
        # split_dataset_df.at[index, 'lemma_sentence'] = lemmas
        return song_df

    @staticmethod
    def main_start(dataset_df):
        dataset_df['stem_data'] = " "
        pool = ThreadPool(GlobalVariable.processor_number)
        result = pool.map(TextualClean.__preprocessing_apply,
                          TextualClean.concat_fields(dataset_df))
        pool.close()
        pool.join()
        return pd.concat(result, sort=False)

    @staticmethod
    def concat_fields(dataset_df):
        dataset_df['data'] = dataset_df['title'] + ' ' + dataset_df['album'] \
                                        + ' ' + dataset_df['artist'] + ' ' + dataset_df['gender']  \
                                        + ' ' + dataset_df['year']
        dataset_df.drop(['title', 'album', 'artist', 'year', 'gender'], inplace=True, axis=1)
        return dataset_df
