import logging
import os

import gc
import pandas as pd

from src.data_models.tdidf_model import FrequencyModel
from src.evaluations.statisticalOverview import StatisticalOverview
from src.globalVariable import GlobalVariable
from src.kemures.tecnics.content_based import ContentBased
from src.preprocessing.preprocessing import Preprocessing

def execute_by_scenarios():
    Preprocessing.database_evaluate_graph()
    application_class_df = pd.DataFrame()
    application_results_df = pd.DataFrame()
    for scenario in GlobalVariable.SCENARIO_SIZE_LIST:
        gc.collect()
        scenario_class_df = pd.DataFrame()
        scenario_results_df = pd.DataFrame()
        for run in range(GlobalVariable.RUN_TIMES):
            os.system('cls||clear')
            logger.info("+ Rodada " + str(run + 1))
            logger.info("+ Carregando o Cenário com " + str(scenario))
            songs_base_df, users_preference_base_df = Preprocessing.load_data_test(scenario)
            run_class_df, run_results_df = ContentBased.run_recommenders(
                users_preference_base_df, FrequencyModel.mold(songs_base_df), scenario, run + 1
            )
            scenario_results_df = pd.concat([scenario_results_df, run_results_df])
            scenario_class_df = pd.concat([scenario_class_df, run_class_df])
        StatisticalOverview.result_info(scenario_results_df)
        StatisticalOverview.graphics(scenario_results_df)
        os.system('cls||clear')
        StatisticalOverview.comparate(scenario_results_df)
        application_results_df = pd.concat([scenario_results_df, application_results_df])
        application_class_df = pd.concat([scenario_class_df, application_class_df])
    StatisticalOverview.scenario_compare(application_results_df)


if __name__ == '__main__':
    GlobalVariable.setup_logging()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.data()
    # SONGS_DF, USERS_PREFERENCES_DF = Preprocessing.load_data()
    logger = logging.getLogger(__name__)
    execute_by_scenarios()
