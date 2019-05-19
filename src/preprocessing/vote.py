import logging
from multiprocessing import Pool

import pandas as pd

from src.globalVariable import GlobalVariable


class Vote:
    def __init__(self, user_preference_df):
        self.user_preference_df = user_preference_df

    @staticmethod
    def user_votate(row, user_std):
        result = True if row['play_count'] >= user_std else False
        return result

    @staticmethod
    def user_vote(user_id, user_preference):
        user_std = user_preference['play_count'].median()
        user_preference['like'] = user_preference.apply(Vote.user_votate, args={user_std}, axis=1)
        return user_preference

    @staticmethod
    def map_user(users_preference_df):
        pass

    def main_start(self):
        logging.info("Iniciando o processo de votação...")
        self.user_preference_df['like'] = None
        split_preference = [(user_id, self.user_preference_df[self.user_preference_df['user_id'] == user_id]) for
                            user_id in self.user_preference_df['user_id'].unique().tolist()]
        pool = Pool(GlobalVariable.processor_number)
        result_df = pool.starmap(Vote.user_vote,
                                 split_preference)
        pool.close()
        pool.join()
        logging.info("Votação finalizada!")
        logging.info("Realizando merge nos resultados...")
        return pd.concat(result_df, sort=False)
