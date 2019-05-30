import json
import logging
import logging.config
import multiprocessing
import os


class GlobalVariable:
    processor_number = multiprocessing.cpu_count() - 1
    execution_times = 5
    user_min_song_list = 20
    RECOMMENDATION_LIST_SIZE = 15
    test_set_size = 0.20
    path_logs = 'logs/logging.json'
    k_neighbors = 3
    song_sample_number = 10000
    sample_random_state = 50
    results_column_name = ['round', 'scenario', 'algorithm', 'metric', 'at', 'value']
    AT_SIZE_LIST = [1, 3, 5, 7, 9, 11, 13, 15]
    SCENARIO_SIZE_LIST = [10000, 30000, 50000]
    GRAPH_STYLE = [':', '--', ':', '-', '-', '-', '--', ':', '--', '-.', '-.']
    GRAPH_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan', '#0F0F0F0F']
    GRAPH_MAKERS = ['o', '^', 's', 'D', 'x', 'p', '.', '1', '|', '*', '2']

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
