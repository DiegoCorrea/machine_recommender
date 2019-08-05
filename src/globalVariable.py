import json
import logging
import logging.config
import multiprocessing
import os


class GlobalVariable:
    processor_number = multiprocessing.cpu_count() - 1
    user_min_song_list = 30
    test_set_size = 0.20
    path_logs = 'logs/logging.json'
    k_neighbors = 3
    sample_random_state = 50
    results_column_name = ['round', 'scenario', 'algorithm', 'metric', 'at', 'value']
    RUN_TIMES = 5
    ONE_SCENARIO_SIZE = 90000
    RECOMMENDATION_LIST_SIZE = 15
    AT_SIZE_LIST = [1, 3, 5, 7, 9, 11, 13, 15]
    SCENARIO_SIZE_LIST = [25000, 50000, 75000]
    GRAPH_STYLE = [':', '--', ':', '-', '-', '-', '--', ':', '--', '-.', '-.']
    GRAPH_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan', '#0F0F0F0F']
    GRAPH_MAKERS = ['o', '^', 's', 'D', 'x', 'p', '.', '1', '|', '*', '2']
    RESULTS_PATH = os.getcwd() + '/results/'

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
