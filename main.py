import logging.config

import pandas as pd

from src.globalVariable import GlobalVariable
from src.models.tdidf_model import FrequencyModel
from src.preprocessing.preprocessing import Preprocessing
from src.evaluations.statisticalOverview import StatisticalOverview

if __name__ == '__main__':
    GlobalVariable.setup_logging()
    # SONGS_DF, USERS_PREFERENCES_DF = preprocessing()
    # SONGS_DF, USERS_PREFERENCES_DF = load_data()
    SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data_test()
    freq_model = FrequencyModel.mold(SONGS_DF)
    results_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    class_balance_check = pd.DataFrame(data=[], columns=['round', 'positive', 'negative'])
    for i in range(GlobalVariable.execution_times):
        for user_id in USERS_PREFERENCES_DF.user_id.unique().tolist():
            user_preference = USERS_PREFERENCES_DF[USERS_PREFERENCES_DF['user_id'] == user_id]
            balance_check = pd.concat(
                [
                    class_balance_check,
                    pd.DataFrame(
                        data=[[i,
                              user_preference[user_preference['like'] == True].shape[0],
                              user_preference[user_preference['like'] == False].shape[0]
                        ]],
                        columns=['round', 'positive', 'negative']
                    )
                ]
            )
    StatisticalOverview.song_info(SONGS_DF)
    StatisticalOverview.song_info(USERS_PREFERENCES_DF)
    StatisticalOverview.song_info(freq_model)
    StatisticalOverview.class_balance_check(class_balance_check)
