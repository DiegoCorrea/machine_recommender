import logging


class StatisticalOverview:
    @staticmethod
    def song_info(df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        df.info(memory_usage='deep')
        logging.info("\n" + str(df.head(5)))
        # logging.info('+ + Total de músicas: ' + str(df.song_id.size))
        # logging.info('+ + Total de Títulos: ' + str(df.title.nunique().count()))
        # logging.info('+ + Total de Artistas: ' + str(df.artist.nunique().count()))
        # logging.info('+ + Total de Albuns: ' + str(df.album.nunique().count()))
        # logging.info('+ + Total de Anos: ' + str(df.year.nunique().count()))
        # logging.info('+ + Total de Gêneros: ' + str(df.gender.nunique().count()))
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
