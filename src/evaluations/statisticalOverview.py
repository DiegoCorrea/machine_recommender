import logging

import matplotlib.pyplot as plt

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
    def tfidf_info(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        df.info(memory_usage='deep')
        logging.info("\n" + str(df.head(5)))
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
    def class_balance_check(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        logging.info("\n" + str(df.tail(5)))
        logging.info("Positive: " + str(df['positive'].sum()))
        logging.info("Negative: " + str(df['negative'].sum()))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def graphics(results_df):
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
                    'results/'
                    + str(scenario)
                    + '_'
                    + metric
                    + '.png',
                    format='png',
                    dpi=300,
                    quality=100,
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight'
                )
                plt.close()

    @staticmethod
    def comparate(results_df):
        """
        :param results_df: Pandas DataFrame com seis colunas: ['round', 'algorithm', 'metric', 'value']
        """
        for scenario in GlobalVariable.SCENARIO_SIZE_LIST:
            # Para cada metrica usada durante a validação dos algoritmos
            print("+ Cenário: ", str(scenario))
            for metric in results_df['metric'].unique().tolist():
                print("+ + Métrica: ", str(metric))
                results_df_by_filter = results_df[
                    (results_df['scenario'] == scenario) &
                    (results_df['metric'] == metric)]
                # Para cada algoritmo usado
                for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                    at_df = results_df[
                        (results_df['scenario'] == scenario) &
                        (results_df['algorithm'] == algorithm) &
                        (results_df['metric'] == metric)
                        ]
                    print("+ + + Algorithm: ", str(algorithm), " -> ",
                          str(at_df['value'].sum() / at_df['value'].count()))
