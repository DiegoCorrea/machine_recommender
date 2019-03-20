import os

import pandas as pd


class DataMining:
    raw_data_path = os.getcwd() + "/data/original_set/"
    clean_data_path = os.getcwd() + "/data/clean_set/"

    def __init__(self):
        self.__song_df = None
        self.__users_preferences_df = None

    def linked_songs(self):
        gender_df = pd.read_csv(DataMining.raw_data_path + 'msd-MAGD-genreAssignment.cls',
                                sep='\t', names=['track_id', 'gender'])
        song_msd_df = pd.read_csv(DataMining.raw_data_path + 'songs.csv',
                                  names=['song_id', 'title', 'album', 'artist', 'year'],
                                  dtype='unicode')
        song_by_track_df = pd.read_csv(DataMining.raw_data_path + 'unique_tracks.txt', engine='python',
                                       sep='<SEP>', names=['track_id', 'song_id', 'title', 'artist'])
        to_merge_df = song_by_track_df.drop(['title', 'artist'], axis=1)
        song_msd_df = song_msd_df.drop_duplicates(['song_id'])
        song_msd_df.info(memory_usage='deep')
        # Join
        self.__song_df = pd.merge(pd.merge(song_msd_df, to_merge_df, how='left', left_on='song_id', right_on='song_id'),
                                  gender_df, how='inner', left_on='track_id', right_on='track_id')
        self.__song_df = self.__song_df.drop_duplicates(['song_id'])
        self.__song_df.set_index("track_id", inplace=True)
        self.__song_df = self.__song_df[str(self.__song_df.year) != "0" or int(self.__song_df.year) != 0 or self.__song_df.year != '0']

    def filter_data_users_by_songs(self):
        users_preferences_df = pd.read_csv(
            DataMining.raw_data_path + 'train_triplets.txt',
            sep='\t', names=['user_id', 'song_id', 'play_count']
        )
        users_preferences_df.info(memory_usage='deep')
        print(str(users_preferences_df.user_id.nunique()))
        self.__users_preferences_df = users_preferences_df[
            users_preferences_df['song_id'].isin(self.__song_df['song_id'].tolist())
        ]

    def save(self):
        self.__song_df.to_csv(DataMining.clean_data_path + 'songs.csv')
        self.__users_preferences_df.to_csv(DataMining.clean_data_path + 'play_count.csv')

    def get_song_df(self):
        return self.__song_df

    def get_user_preferences_df(self):
        return self.__users_preferences_df

    @staticmethod
    def create():
        pre = DataMining()
        pre.linked_songs()
        pre.filter_data_users_by_songs()
        pre.save()
        return pre.get_song_df(), pre.get_user_preferences_df()

    @staticmethod
    def update_songs(song_df):
        song_df.to_csv(DataMining.clean_data_path + 'songs.csv')

    @staticmethod
    def load_song_set():
        song_df = pd.read_csv(DataMining.clean_data_path + 'songs.csv')
        return song_df.set_index("track_id")

    @staticmethod
    def load_user_set():
        return pd.read_csv(DataMining.clean_data_path + 'play_count.csv')
