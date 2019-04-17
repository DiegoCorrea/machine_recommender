import multiprocessing
import logging
import logging.config
import json
import os


class GlobalVariable:
    processor_number = multiprocessing.cpu_count() - 1
    execution_times = 3
    user_min_song_list = 10
    path_logs = 'logs/logging.json'

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