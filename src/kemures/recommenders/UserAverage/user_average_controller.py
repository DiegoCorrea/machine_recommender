# -*- coding: utf-8 -*-
import pandas as pd

from src.globalVariable import GlobalVariable


class UserAverageController:
    @staticmethod
    def start_ranking(similarity_data_df, user_model_ids, song_model_ids):
        user_model_df = similarity_data_df.loc[user_model_ids]
        song_model_df = similarity_data_df.loc[song_model_ids]
        song_model_df = song_model_df.drop(columns=song_model_ids)
        recommendation_list = {}
        for column in song_model_df.columns:
            column_values = song_model_df[column].values.tolist()
            column_values = [i for i in column_values if i != 0.0]
            if len(column_values) == 0:
                continue
            similarity = float(sum(column_values)) / float(
                len(user_model_df.index))
            if similarity == 0.0:
                continue
            recommendation_list[column] = [similarity]
        user_recommendations_df = pd.DataFrame.from_dict(data=dict(recommendation_list), orient='index',
                                                         columns=['similarity'])
        resp_user_recommendation_df = user_recommendations_df.sort_values(by=['similarity'], ascending=False).iloc[
                                      0:GlobalVariable.RECOMMENDATION_LIST_SIZE]
        return resp_user_recommendation_df
