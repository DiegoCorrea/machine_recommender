import logging

import pandas as pd

from src.evaluations.validation import Validation
from src.globalVariable import GlobalVariable
from src.machine_learning.sklearn_config import MachineAlgorithms


class ContentBased:
    logger = logging.getLogger(__name__)
    @staticmethod
    def run_recommenders(users_dataset_df, freq_model, scenario, run):
        class_balance_check = pd.DataFrame(data=[], columns=['round', 'positive', 'negative'])
        users_results_df = pd.DataFrame(data=[], columns=GlobalVariable.results_column_name)
        ContentBased.logger.info("Total de usuários nesta rodada: " + str(users_dataset_df.user_id.nunique()))
        for user_id in users_dataset_df.user_id.unique().tolist():
            user_preference = users_dataset_df[users_dataset_df['user_id'] == user_id]
            if user_preference['like'].nunique() == 1 or user_preference[user_preference['like'] == True].shape[0] < 2 or user_preference[user_preference['like'] == False].shape[0] < 2:
                continue
            class_balance_check = pd.concat(
                [
                    class_balance_check,
                    pd.DataFrame(
                        data=[[run,
                               user_preference[user_preference['like'] == True].shape[0],
                               user_preference[user_preference['like'] == False].shape[0]
                               ]],
                        columns=['round', 'positive', 'negative']
                    )
                ]
            )
            x_train_data, x_test_data, y_train_label, y_test_label = Validation.split_data(
                user_preference['song_id'].values.tolist(), user_preference['like'].values.tolist())
            users_results_df = pd.concat(
                [
                    users_results_df,
                    MachineAlgorithms.main(
                        x_train=freq_model.loc[x_train_data],
                        x_test=freq_model.loc[x_test_data],
                        y_train=y_train_label,
                        y_test=y_test_label,
                        run=run,
                        scenario=scenario
                    )
                ]
            )
        # ####
        results_df = ContentBased.users_result_generate(users_results_df, run, scenario)
        return class_balance_check, results_df

    @staticmethod
    def users_result_generate(users_results_df, run, scenario):
        result_df = pd.DataFrame(data=[], columns=GlobalVariable.results_column_name)
        for algorithm in users_results_df['algorithm'].unique().tolist():
            algorithm_subset = users_results_df[users_results_df['algorithm'] == algorithm]
            for metric in algorithm_subset['metric'].unique().tolist():
                metric_subset = algorithm_subset[algorithm_subset['metric'] == metric]
                for at in metric_subset['at'].unique().tolist():
                    at_subset = metric_subset[metric_subset['at'] == at]
                    result_df = pd.concat([
                        result_df,
                        pd.DataFrame(data=[[run, scenario, algorithm, metric, at, at_subset['value'].mean()]],
                                     columns=GlobalVariable.results_column_name)
                    ])
        return result_df
