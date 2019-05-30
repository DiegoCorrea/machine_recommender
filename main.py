import logging
import os

from src.data_models.tdidf_model import FrequencyModel
from src.evaluations.statisticalOverview import StatisticalOverview
from src.globalVariable import GlobalVariable
from src.kemures.tecnics.content_based import ContentBased
from src.preprocessing.preprocessing import Preprocessing

if __name__ == '__main__':
    GlobalVariable.setup_logging()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.data()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data()
    logger = logging.getLogger(__name__)
    for scenario in GlobalVariable.SCENARIO_SIZE_LIST:
        os.system('cls||clear')
        logger.info("+ Carregando o Cenário com " + str(scenario))
        SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data_test(scenario)
        freq_model = FrequencyModel.mold(SONGS_DF)
        class_balance_check, results_df = ContentBased.run_recommenders(USERS_PREFERENCES_DF, freq_model, scenario)
        StatisticalOverview.class_balance_check(class_balance_check)
        StatisticalOverview.song_info(SONGS_DF)
        StatisticalOverview.user_info(USERS_PREFERENCES_DF)
        StatisticalOverview.tfidf_info(freq_model)
        StatisticalOverview.result_info(results_df)
        StatisticalOverview.graphics(results_df)
