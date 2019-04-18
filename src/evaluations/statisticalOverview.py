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
        # Para cada metrica usada durante a validação dos algoritmos
        for metric in results_df['metric'].unique().tolist():
            # Cria e configura gráficos
            plt.figure()
            plt.grid(True)
            plt.xlabel('Rodada')
            plt.ylabel('Valor')
            results_df_by_filter = results_df[results_df['metric'] == metric]
            # Para cada algoritmo usado cria-se uma linha no gráfico com cores e formatos diferentes
            n = results_df_by_filter['algorithm'].nunique()
            for algorithm, style, colors, makers in zip(results_df_by_filter['algorithm'].unique().tolist(),
                                                        GlobalVariable.GRAPH_STYLE[:n],
                                                        GlobalVariable.GRAPH_COLORS[:n],
                                                        GlobalVariable.GRAPH_MAKERS[:n]):
                at_df = results_df[
                    (results_df['algorithm'] == algorithm) &
                    (results_df['metric'] == metric)]
                at_df.sort_values("round")
                plt.plot(
                    at_df['round'],
                    at_df['value'],
                    linestyle=style,
                    color=colors,
                    marker=makers,
                    label=algorithm
                )
            # Configura legenda
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.xticks(sorted(results_df['round'].unique().tolist()))
            # Salva a figura com alta resolução e qualidade
            plt.savefig(
                'results/'
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
        for config in results_df['config'].unique().tolist():
            # Para cada modelagem de dados
            for model in results_df['model'].unique().tolist():
                # Para cada metrica usada durante a validação dos algoritmos
                for metric in results_df['metric'].unique().tolist():
                    results_df_by_filter = results_df[
                        (results_df['config'] == config) &
                        (results_df['model'] == model) &
                        (results_df['metric'] == metric)]
                    # Para cada algoritmo usado
                    for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                        at_df = results_df[
                            (results_df['config'] == config) &
                            (results_df['algorithm'] == algorithm) &
                            (results_df['model'] == model) &
                            (results_df['metric'] == metric)]
                        print("Config; ", str(config), "\t| Algoritmo: ", str(algorithm), "\t| Model: ", str(model),
                              "\t| Metrica: ", str(metric), "RESULT; ",
                              str(at_df['value'].sum() / at_df['value'].count()))
