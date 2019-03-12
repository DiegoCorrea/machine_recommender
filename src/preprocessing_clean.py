import pandas as pd
import os


class PreProcessing:
    def __init__(self):
        self.__raw_data_path = "data/original_set/"
        self.__clean_data_path = "data/clean_set/"
        self.__song_df = None

    @classmethod
    def clean(cls):
        pass

    def link_songs(self):
        gender_df = pd.read_csv(os.getcwd() + '/data/original_set/msd-MAGD-genreAssignment.cls',
                                sep='\t', names=['track_id', 'gender'])
        song_msd_df = pd.read_csv(os.getcwd() + '/data/original_set/songs.csv',
                                  names=['id', 'title', 'album', 'artist', 'year'])
        song_by_track_df = pd.read_csv(os.getcwd() + '/data/original_set/unique_tracks.txt',
                                       sep='<SEP>', names=['track_id', 'id', 'title', 'artist'])
        to_merge_df = song_by_track_df.drop(['title', 'artist'], axis=1)
        # song_by_track_df = self.__song_by_track_df.drop_duplicates(['id'])
        # Join
        self.__song_df = pd.merge(pd.merge(song_msd_df, to_merge_df, how='left', left_on='id', right_on='id'),
                                  gender_df, how='inner', left_on='track_id', right_on='track_id')

    def get_songs(self):
        return self.__song_df
