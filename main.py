import pandas as pd
from src.preprocessing.caller import preprocessing, load_data, load_data_test
from src.models.tdidf_model import FrequencyModel
from src.globalVariable import GlobalVariable
from src.preprocessing.vote import Vote

import logging
import logging.config
import json
import os


def setup_logging(
    default_path='logs/logging.json',
    default_level=logging.DEBUG,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        config = {}
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    setup_logging()
    # SONGS_DF, USERS_PREFERENCES_DF = preprocessing()
    # SONGS_DF, USERS_PREFERENCES_DF = load_data()
    SONGS_DF, USERS_PREFERENCES_DF = load_data_test()
    logging.info("*" * 50)
    logging.info("*" * 50)
    SONGS_DF.info(memory_usage='deep')
    logging.info(SONGS_DF.head(5))
    logging.info("*" * 50)
    logging.info("*" * 50)
    USERS_PREFERENCES_DF.info(memory_usage='deep')
    logging.info(USERS_PREFERENCES_DF.head(5))
    logging.info("*" * 50)
    logging.info("*" * 50)
    freq_model = FrequencyModel.mold(SONGS_DF)
    freq_model.info(memory_usage='deep')
    logging.info(freq_model.head(5))
    results_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    vote = Vote(USERS_PREFERENCES_DF)
    USERS_PREFERENCES_DF = vote.main_start()
    for i in range(GlobalVariable.execution_times):
        for user_id in USERS_PREFERENCES_DF.user_id.unique().tolist():
            user_preference = USERS_PREFERENCES_DF[USERS_PREFERENCES_DF['user_id'] == user_id]
            # print(str(len(user_preference)))

