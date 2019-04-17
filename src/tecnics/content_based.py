import logging

import pandas as pd

from src.algorithms.machine_algorithms import MachineAlgorithms
from src.evaluations.validation import Validation
from src.globalVariable import GlobalVariable


class ContentBased:
    @staticmethod
    def run_recommenders(users_dataset_df, freq_model):
        class_balance_check = pd.DataFrame(data=[], columns=['round', 'positive', 'negative'])
        results_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
        for i in range(GlobalVariable.execution_times):
            logging.info("Rodada " + str(i))
            for user_id in users_dataset_df.user_id.unique().tolist():
                user_preference = users_dataset_df[users_dataset_df['user_id'] == user_id]
                class_balance_check = pd.concat(
                    [
                        class_balance_check,
                        pd.DataFrame(
                            data=[[i,
                                   user_preference[user_preference['like'] == True].shape[0],
                                   user_preference[user_preference['like'] == False].shape[0]
                                   ]],
                            columns=['round', 'positive', 'negative']
                        )
                    ]
                )
                x_train_data, x_test_data, y_train_label, y_test_label = Validation.split_data(
                    user_preference['song_id'].values.tolist(), user_preference['like'].values.tolist())
                results_df = pd.concat(
                    [
                        results_df,
                        MachineAlgorithms.main(
                            x_train=freq_model.loc[x_train_data],
                            x_test=freq_model.loc[x_test_data],
                            y_train=y_train_label,
                            y_test=y_test_label,
                            run=i,
                            model='TF-IDF'
                        )
                    ]
                )
        return class_balance_check, results_df
