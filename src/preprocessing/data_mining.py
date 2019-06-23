import logging
import os

import pandas as pd

from src.globalVariable import GlobalVariable
from src.preprocessing.base_analytics import BaseAnalytics
from src.preprocessing.vote import Vote


class DataMining:
    raw_data_path = os.getcwd() + "/data/original_set/"
    clean_data_path = os.getcwd() + "/data/clean_set/"
    inner_join_data_path = os.getcwd() + "/data/inner_join_set/"
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.__song_df = pd.DataFrame()
        self.__users_preferences_df = pd.DataFrame()

    def get_song_df(self):
        return self.__song_df

    def get_user_preferences_df(self):
        return self.__users_preferences_df

    def create_song_dataset(self):
        song_msd_df = DataMining.load_raw_songs()
        print("#" * 50)
        print("1 - Base de Dados Musicais Original")
        print("#" * 50)
        BaseAnalytics.song_info(song_msd_df)
        song_msd_df = song_msd_df.drop_duplicates(['song_id'])
        print("#" * 50)
        print("2 - Base de Dados Musicais Deletando as instâncias duplicadas")
        print("#" * 50)
        BaseAnalytics.song_info(song_msd_df)
        genre_df = DataMining.load_raw_gender()
        print("#" * 50)
        print("3 - Base de Dados bruta gêneros")
        print("#" * 50)
        BaseAnalytics.raw_genre_info(genre_df)
        self.__song_df = pd.merge(
            pd.merge(song_msd_df, DataMining.load_raw_track(),
                     how='left', left_on='song_id', right_on='song_id'),
            genre_df, how='inner', left_on='track_id', right_on='track_id')
        self.__song_df = self.__song_df.drop_duplicates(['song_id'])
        self.__song_df.set_index("track_id", inplace=True, drop=True)
        print("#" * 50)
        print("4 - Base de Dados com o merge de gêneros")
        print("#" * 50)
        BaseAnalytics.song_complete_info(self.__song_df)
        indexNames = self.__song_df[(self.__song_df['year'] == '0')].index
        self.__song_df.drop(indexNames, inplace=True)
        print("#" * 50)
        print("5 - Base de Dados sem o ano zerados")
        print("#" * 50)
        BaseAnalytics.song_complete_info(self.__song_df)

    def filter_data_users_by_songs(self):
        users_preferences_df = pd.read_csv(
            DataMining.raw_data_path + 'train_triplets.txt',
            sep='\t', names=['user_id', 'song_id', 'play_count']
        )
        print("-" * 50)
        print("1 - Base de Dados bruta das preferências")
        print("-" * 50)
        BaseAnalytics.user_info(users_preferences_df)
        self.__users_preferences_df = users_preferences_df[
            users_preferences_df['song_id'].isin(self.__song_df['song_id'].tolist())
        ]
        print("-" * 50)
        print("2 - Base de Dados dos usuários filtrada")
        print("-" * 50)
        BaseAnalytics.user_info(self.__users_preferences_df)

    def filter_heard_songs(self):
        self.__song_df = self.__song_df[
            self.__song_df['song_id'].isin(self.__users_preferences_df['song_id'].unique().tolist())
        ]
        print("%" * 50)
        print("1 - Base de Dados musicais final!")
        print("%" * 50)
        BaseAnalytics.song_complete_info(self.__song_df)

    @staticmethod
    def create():
        dm = DataMining()
        logging.info("Músicas: Concatenando bases.")
        dm.create_song_dataset()
        logging.info("Filtrando usuários e músicas.")
        dm.filter_data_users_by_songs()
        dm.filter_heard_songs()
        logging.info("Salvando...")
        dm.save_intermediate()
        return dm.get_song_df(), dm.get_user_preferences_df()

    @staticmethod
    def update(song_df, user_set):
        song_df.to_csv(DataMining.clean_data_path + 'songs.csv')
        user_set.to_csv(DataMining.clean_data_path + 'play_count.csv', index=False)

    @staticmethod
    def load_song_set():
        song_df = pd.read_csv(DataMining.clean_data_path + 'songs.csv')
        return song_df.set_index("track_id")

    @staticmethod
    def load_user_set():
        return pd.read_csv(DataMining.clean_data_path + 'play_count.csv')

    @staticmethod
    def load_inner_song_set():
        song_df = pd.read_csv(DataMining.clean_data_path + 'songs.csv')
        return song_df.set_index("track_id")

    @staticmethod
    def load_inner_user_set():
        return pd.read_csv(DataMining.clean_data_path + 'play_count.csv')

    @staticmethod
    def load_raw_gender():
        return pd.read_csv(DataMining.raw_data_path + 'msd-MAGD-genreAssignment.cls',
                           sep='\t', names=['track_id', 'genre'])

    @staticmethod
    def load_raw_songs():
        return pd.read_csv(DataMining.raw_data_path + 'songs.csv',
                           names=['song_id', 'title', 'album', 'artist', 'year'],
                           dtype='unicode')

    @staticmethod
    def load_raw_track():
        song_by_track_df = pd.read_csv(DataMining.raw_data_path + 'unique_tracks.txt', engine='python',
                                       sep='<SEP>', names=['track_id', 'song_id', 'title', 'artist'])
        return song_by_track_df.drop(['title', 'artist'], axis=1)

    def save_intermediate(self):
        self.__song_df.to_csv(DataMining.inner_join_data_path + 'songs.csv', header=True)
        self.__users_preferences_df.to_csv(DataMining.inner_join_data_path + 'play_count.csv', index=False)

    def save(self):
        self.__song_df.to_csv(DataMining.clean_data_path + 'songs.csv', header=True)
        self.__users_preferences_df.to_csv(DataMining.clean_data_path + 'play_count.csv', index=False)

    @staticmethod
    def load_set_test(scenario_size):
        DataMining.logger.info("Carregando músicas e realizando sample...")
        song_df = pd.read_csv(DataMining.clean_data_path + 'songs.csv')
        song_df.set_index("track_id", inplace=True)
        song_sample = song_df.sample(n=scenario_size)
        # load users
        DataMining.logger.info("Carregando usuários...")
        users_preferences_df = pd.read_csv(DataMining.clean_data_path + 'play_count.csv')
        DataMining.logger.info("Filtrando usuários...")
        user_sample = users_preferences_df[
            users_preferences_df['song_id'].isin(song_sample['song_id'].tolist())
        ]
        result = user_sample.user_id.value_counts()
        select_user = [a for a, b in result.iteritems() if (b >= GlobalVariable.user_min_song_list)]
        user_sample = user_sample[
            user_sample['user_id'].isin(select_user)
        ]
        DataMining.logger.info("Realizando votação do usuário...")
        vote = Vote(user_sample)
        user_sample = vote.main_start()
        return song_sample, user_sample
