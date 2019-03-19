import os

import pandas as pd


class DataMining:
    def __init__(self):
        self.__raw_data_path = os.getcwd() + "/data/original_set/"
        self.__clean_data_path = os.getcwd() + "/data/clean_set/"
        self.__song_df = None
        self.__users_preferences_df = None

    def linked_songs(self):
        gender_df = pd.read_csv(self.__raw_data_path + 'msd-MAGD-genreAssignment.cls',
                                sep='\t', names=['track_id', 'gender'])
        song_msd_df = pd.read_csv(self.__raw_data_path + 'songs.csv',
                                  names=['id', 'title', 'album', 'artist', 'year'])
        song_by_track_df = pd.read_csv(self.__raw_data_path + 'unique_tracks.txt',
                                       sep='<SEP>', names=['track_id', 'id', 'title', 'artist'])
        to_merge_df = song_by_track_df.drop(['title', 'artist'], axis=1)
        song_msd_df = song_msd_df.drop_duplicates(['id'])
        # Join
        self.__song_df = pd.merge(pd.merge(song_msd_df, to_merge_df, how='left', left_on='id', right_on='id'),
                                  gender_df, how='inner', left_on='track_id', right_on='track_id')

    def filter_data_users_by_songs(self):
        users_preferences_df = pd.read_csv(
            self.__raw_data_path + 'train_triplets.txt',
            sep='\t', names=['user_id', 'song_id', 'play_count']
        )
        users_preferences_df.info(verbose=True)
        self.__users_preferences_df = users_preferences_df[
            users_preferences_df['song_id'].isin(self.__song_df['id'].tolist())
        ]
        self.__users_preferences_df.info(verbose=True)

    def save(self):
        self.__song_df.to_csv(self.__clean_data_path + 'songs.csv', index=False,
                              columns=['id', 'title', 'artist', 'album', 'gender'])
        self.__users_preferences_df.to_csv(self.__clean_data_path + 'play_count.csv', index=False,
                                           columns=['user_id', 'song_id', 'play_count'])

    def get_song_df(self):
        return self.__song_df

    def get_user_preferences_df(self):
        return self.__users_preferences_df
