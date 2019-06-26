import logging

from .data_mining import DataMining
from .textual_clean import TextualClean


class Preprocessing:
    @staticmethod
    def data():
        # Create Dataset
        logging.info("Iniciando a criação da base de dados.")
        song_set, user_set = DataMining.create()
        logging.info("*" * 50)
        logging.info("Limpando os dados textuais.")
        songs_df = TextualClean.main_start(song_set)
        logging.info("Atualizando arquivos do dataset.")
        DataMining.update(songs_df, user_set)
        return songs_df, user_set

    @staticmethod
    def load_data():
        return DataMining.load_song_set(), DataMining.load_user_set()

    @staticmethod
    def load_data_test(scenario):
        return DataMining.load_set_test(scenario)