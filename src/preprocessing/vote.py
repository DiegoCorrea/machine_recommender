import logging
from multiprocessing import Pool

import pandas as pd

from src.globalVariable import GlobalVariable


class Vote:
    logger = logging.getLogger(__name__)

    def __init__(self, user_preference_df):
        self.user_preference_df = user_preference_df

    @staticmethod
    def user_votate(row, user_std):
        result = True if row['play_count'] >= user_std else False
        return result

    @staticmethod
    def user_vote(user_preference):
        user_std = user_preference['play_count'].median()
        user_preference['like'] = user_preference.apply(Vote.user_votate, args={user_std}, axis=1)
        return user_preference

    @staticmethod
    def map_user(users_preference_df):
        pass

    def main_start(self):
        Vote.logger.info("Iniciando o processo de votação...")
        self.user_preference_df['like'] = None
        gb = self.user_preference_df.groupby('user_id')
        # split_preference = [(user_id, self.user_preference_df[self.user_preference_df['user_id'] == user_id]) for
        #                     user_id in self.user_preference_df['user_id'].unique().tolist()]
        Vote.logger.info("Grupos")
        groups = [gb.get_group(x) for x in gb.groups]
        Vote.logger.info("Abrindo Threads")
        pool = Pool(GlobalVariable.processor_number)
        result_df = pool.map(Vote.user_vote, groups)
        pool.close()
        pool.join()
        Vote.logger.info("Votação finalizada!")
        Vote.logger.info("Realizando merge nos resultados...")
        return pd.concat(result_df, sort=False)
