import json
import logging
import logging.config
import multiprocessing
import os


class GlobalVariable:
    processor_number = multiprocessing.cpu_count() - 1
    execution_times = 3
    user_min_song_list = 10
    test_set_size = 0.20
    path_logs = 'logs/logging.json'
    k_neighbors = 3
    song_sample_number = 10000
    sample_random_state = 50

    @staticmethod
    def setup_logging(
            default_path=path_logs,
            default_level=logging.DEBUG,
            env_key='LOG_CFG'
    ):
        """
            Setup logging configuration
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
