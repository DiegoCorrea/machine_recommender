from src.evaluations.statisticalOverview import StatisticalOverview
from src.globalVariable import GlobalVariable
from src.models.tdidf_model import FrequencyModel
from src.preprocessing.preprocessing import Preprocessing
from src.tecnics.content_based import ContentBased

if __name__ == '__main__':
    GlobalVariable.setup_logging()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.data()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data()
    SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data_test()
    freq_model = FrequencyModel.mold(SONGS_DF)
    class_balance_check, results_df = ContentBased.run_recommenders(USERS_PREFERENCES_DF, freq_model)
    StatisticalOverview.song_info(SONGS_DF)
    StatisticalOverview.user_info(USERS_PREFERENCES_DF)
    StatisticalOverview.tfidf_info(freq_model)
    StatisticalOverview.class_balance_check(class_balance_check)
    StatisticalOverview.result_info(results_df)
