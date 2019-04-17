import logging
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd

from src.globalVariable import GlobalVariable


class Vote:
    def __init__(self, user_preference_df):
        self.user_preference_df = user_preference_df

    @staticmethod
    def __user_votate(row, user_std):
        result = True if row['play_count'] >= user_std else False
        return result

    def __user_vote(self, user_id):
        user_preference = self.user_preference_df[self.user_preference_df['user_id'] == user_id]
        user_std = user_preference['play_count'].std()
        user_preference['like'] = user_preference.apply(Vote.__user_votate, args={user_std}, axis=1)
        return user_preference

    def main_start(self):
        logging.info("Iniciando o processo de votação...")
        self.user_preference_df['like'] = None
        pool = ThreadPool(GlobalVariable.processor_number)
        result_df = pool.map(self.__user_vote,
                             self.user_preference_df['user_id'].unique().tolist())
        pool.close()
        pool.join()
        logging.info("Votação finalizada!")
        logging.info("Realizando merge nos resultados...")
        return pd.concat(result_df, sort=False)
