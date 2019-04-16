import pandas as pd
from src.preprocessing.caller import preprocessing, load_data, load_data_test
from src.models.tdidf_model import FrequencyModel
EXECUTION_TIMES = 3


if __name__ == '__main__':
    # SONGS_DF, USERS_PREFERENCES_DF = preprocessing()
    # SONGS_DF, USERS_PREFERENCES_DF = load_data()
    SONGS_DF, USERS_PREFERENCES_DF = load_data_test()
    print("*" * 50)
    print("*" * 50)
    SONGS_DF.info(memory_usage='deep')
    SONGS_DF.head()
    print("*" * 50)
    print("*" * 50)
    USERS_PREFERENCES_DF.info(memory_usage='deep')
    USERS_PREFERENCES_DF.head()
    print(str(USERS_PREFERENCES_DF.song_id.nunique()))
    print("*" * 50)
    print("*" * 50)
    freq_model = FrequencyModel.mold(SONGS_DF)
    freq_model.info(memory_usage='deep')
    freq_model.head()
    results_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    for i in range(EXECUTION_TIMES):
        for user_id in USERS_PREFERENCES_DF.user_id.unique().tolist():
            user_preference = USERS_PREFERENCES_DF.loc[USERS_PREFERENCES_DF['user_id'] == user_id]
            print(str(len(user_preference)))

