import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.globalVariable import GlobalVariable


class StatisticalOverview:
    @staticmethod
    def song_info(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        df.info(memory_usage='deep')
        logging.info("\n" + str(df.head(5)))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def user_info(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        df.info(memory_usage='deep')
        logging.info("\n" + str(df.head(5)))
        logging.info('+ + Total de usuarios: ' + str(df.user_id.size))
        logging.info('+ + Total de músicas ouvidas: ' + str(df.song_id.nunique()))
        logging.info('+ + Música mais ouvida: ' + str(df.play_count.max()))
        logging.info('+ + Música menos ouvida: ' + str(df.play_count.min()))
        logging.info('+ + Total de Reproduções: ' + str(df.play_count.sum()))
        logging.info('+ + Desvio Padrão de Reproduções: ' + str(df.play_count.std()))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def result_info(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        df.info(memory_usage='deep')
        logging.info("\n" + str(df.head(5)))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def print_scenario(results_df, scenario):
        """
        :param scenario:
        :param results_df: Pandas DataFrame com seis colunas: ['round', 'algorithm', 'metric', 'value']
        """
        print("+ Cenário: ", str(scenario))
        for metric in results_df['metric'].unique().tolist():
            print("+ + Métrica: ", str(metric))
            results_df_by_filter = results_df[
                results_df['metric'] == metric]
            # Para cada algoritmo usado
            for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                at_df = results_df_by_filter[
                    results_df_by_filter['algorithm'] == algorithm
                    ]
                print("+ + + Algorithm: ", str(algorithm), " -> ",
                      str(at_df['value'].mean()))

    @staticmethod
    def scenario_graphic(results_df):
        """
        Gera todos os gráficos. Para qualquer modelo e todas as métricas cria um gráfico com os algoritmos nas linhas
        :param results_df: Pandas DataFrame com cinco colunas: ['round', 'algorithm', 'metric', 'value']
        """
        for scenario in results_df['scenario'].unique().tolist():
            # Para cada metrica usada durante a validação dos algoritmos
            results_df_by_scenario = results_df[results_df['scenario'] == scenario]
            for metric in results_df_by_scenario['metric'].unique().tolist():
                # Cria e configura gráficos
                plt.figure()
                plt.figure(figsize=(10, 6))
                plt.grid(True)
                plt.rc('xtick', labelsize=14)
                plt.rc('ytick', labelsize=14)
                plt.xlabel('Tamanho da lista', fontsize=18)
                plt.ylabel('Score', fontsize=18)
                results_df_by_scenario_metric = results_df_by_scenario[results_df_by_scenario['metric'] == metric]
                # Para cada algoritmo usado cria-se uma linha no gráfico com cores e formatos diferentes
                n = results_df_by_scenario_metric['algorithm'].nunique()
                for algorithm, style, colors, makers in zip(
                        results_df_by_scenario_metric['algorithm'].unique().tolist(),
                        GlobalVariable.GRAPH_STYLE[:n],
                        GlobalVariable.GRAPH_COLORS[:n],
                        GlobalVariable.GRAPH_MAKERS[:n]):
                    results_df_by_scenario_metric_algorithm = results_df_by_scenario_metric[
                        results_df_by_scenario_metric['algorithm'] == algorithm]
                    results = dict()
                    results['at'] = []
                    results['value'] = []
                    for at in results_df_by_scenario_metric_algorithm['at'].unique().tolist():
                        at_df = results_df_by_scenario_metric_algorithm[
                            results_df_by_scenario_metric_algorithm['at'] == at]
                        results['at'].append(at)
                        results['value'].append(at_df['value'].mean())
                    print(results)
                    plt.plot(
                        results['at'],
                        results['value'],
                        linestyle=style,
                        color=colors,
                        marker=makers,
                        label=algorithm
                    )
                # Configura legenda
                lgd = plt.legend(loc=9, prop={'size': 18}, bbox_to_anchor=(0.5, -0.1), ncol=3)
                plt.xticks(sorted(GlobalVariable.AT_SIZE_LIST))
                # Salva a figura com alta resolução e qualidade
                plt.savefig(
                    GlobalVariable.RESULTS_PATH
                    + metric
                    + '_'
                    + str(scenario)
                    + '.png',
                    format='png',
                    dpi=300,
                    quality=100,
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight'
                )
                plt.close()

    @staticmethod
    def scenario_compare(df):
        for metric in df['metric'].unique().tolist():
            plt.figure()
            values = []
            labels = []
            print("+ + Métrica: ", str(metric))
            results_df_by_filter = df[df['metric'] == metric]
            for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                at_df = df[
                    (df['algorithm'] == algorithm) &
                    (df['metric'] == metric)
                    ]
                labels.append(algorithm)
                values.append(at_df['value'].mean())
            y_pos = np.arange(len(labels))
            plt.bar(y_pos, values, align='center', alpha=0.5)
            plt.xticks(y_pos, labels)
            plt.grid(axis='y')
            plt.ylabel('Score')
            plt.xticks(rotation=30)
            plt.savefig(
                GlobalVariable.RESULTS_PATH
                + metric
                + '_'
                + 'final_compare_results'
                + '.png',
                format='png',
                dpi=300,
                quality=100
            )
            plt.close()

    @staticmethod
    def save_scenario_as_csv(df, scenario):
        df.to_csv(GlobalVariable.RESULTS_PATH + str(scenario) + ".csv", header=True)

    @staticmethod
    def final_results_as_csv(df):
        for metric in df['metric'].unique().tolist():
            results_df_by_filter = df[df['metric'] == metric]
            results = pd.DataFrame(columns=['results'], index=results_df_by_filter['algorithm'].unique().tolist())
            for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                at_df = df[
                    (df['algorithm'] == algorithm) &
                    (df['metric'] == metric)
                    ]
                results.at[algorithm, 'results'] = at_df['value'].mean()
            results.to_csv(GlobalVariable.RESULTS_PATH + str(metric) + ".csv", header=True)

    @staticmethod
    def final_results(df):
        StatisticalOverview.scenario_compare(df)
        StatisticalOverview.final_results_as_csv(df)

    @staticmethod
    def save_class_results_as_cdv(df, scenario):
        df.to_csv(GlobalVariable.RESULTS_PATH + str(scenario) + "_users" + ".csv", header=True)
