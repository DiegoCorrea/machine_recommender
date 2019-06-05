import logging

import matplotlib.pyplot as plt

from .data_mining import DataMining
from .textual_clean import TextualClean


class Preprocessing:
    @staticmethod
    def data():
        # Create Dataset
        logging.info("Iniciando a criação da base de dados...")
        song_set, user_set = DataMining.create()
        logging.info("*" * 50)
        # Load Dataset
        # song_set = DataMining.load_song_set()
        # user_set = DataMining.load_user_set()
        logging.info("Limpando os dados textuais...")
        song_set.head()
        songs_df = TextualClean.main_start(song_set)
        song_set.head()
        logging.info("Atualizando arquivos do dataset...")
        DataMining.update_songs(songs_df)
        song_set.head()
        return songs_df, user_set

    @staticmethod
    def load_data():
        return DataMining.load_song_set(), DataMining.load_user_set()

    @staticmethod
    def load_data_test(scenario):
        return DataMining.load_set_test(scenario)

    @staticmethod
    def database_evaluate_graph():
        user_preference_base_df = DataMining.load_user_set()
        x = user_preference_base_df.sort_values(by=['play_count'])
        plt.figure()
        plt.xlabel('Preferência do usuário normalizada')
        plt.ylabel('Quantidade')
        data = (x['play_count'].values.tolist() / x['play_count'].max())
        plt.hist(data, bins=100, alpha=0.5,
                 histtype='bar', color='steelblue',
                 edgecolor='black')
        plt.grid(axis='y')
        plt.savefig(
            'results/'
            + 'user_play_count_histo.eps', format='eps', dpi=300
        )
        plt.close()
