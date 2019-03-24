from src.preprocessing.caller import preprocessing, load_data
from src.models.tdidf_model import FrequencyModel


if __name__ == '__main__':
    # SONGS_DF, USER_PREFERENCES_DF = preprocessing()
    SONGS_DF, USER_PREFERENCES_DF = load_data()
    print("*" * 50)
    print("*" * 50)
    SONGS_DF.info(memory_usage='deep')
    SONGS_DF.head()
    print("*" * 50)
    print("*" * 50)
    USER_PREFERENCES_DF.info(memory_usage='deep')
    USER_PREFERENCES_DF.head()
    print(str(USER_PREFERENCES_DF.song_id.nunique()))
    print("*" * 50)
    print("*" * 50)
    freq_model = FrequencyModel.mold(SONGS_DF.head(3000))
    freq_model.info(memory_usage='deep')
    freq_model.head()

