# -*- coding: utf-8 -*-
import pandas as pd

from src.globalVariable import GlobalVariable


class UserAverageController:
    @staticmethod
    def start_ranking(similarity_data_df, user_model_ids, song_model_ids):
        song_model_df = similarity_data_df.loc[song_model_ids]
        song_model_df = song_model_df.drop(columns=song_model_ids)
        recommendation_list = {}
        for song_id in song_model_ids:
            song_values = song_model_df.loc[song_id].values.tolist()
            if len(song_values) == 0:
                recommendation_list[song_id] = [0]
                continue
            similarity = float(sum(song_values)) / float(
                len(user_model_ids))
            if similarity == 0.0:
                continue
            recommendation_list[song_id] = [similarity]
        user_recommendations_df = pd.DataFrame.from_dict(data=dict(recommendation_list), orient='index',
                                                         columns=['similarity'])
        resp_user_recommendation_df = user_recommendations_df.sort_values(by=['similarity'], ascending=False).iloc[
                                      0:GlobalVariable.RECOMMENDATION_LIST_SIZE]
        return resp_user_recommendation_df
