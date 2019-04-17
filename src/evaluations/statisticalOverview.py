import logging.config


class StatisticalOverview:
    @staticmethod
    def song_info(song_df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        song_df.info(memory_usage='deep')
        logging.info("\n" + str(song_df.head(5)))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def user_info(user_df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        user_df.info(memory_usage='deep')
        logging.info("\n" + str(user_df.head(5)))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def tfidf_info(tfidf_df):
        logging.info("*" * 50)
        logging.info("*" * 50)
        tfidf_df.info(memory_usage='deep')
        logging.info("\n" + str(tfidf_df.head(5)))
        logging.info("=" * 50)
        logging.info("=" * 50)

    @staticmethod
    def class_balance_check(balance_check):
        logging.info("*" * 50)
        logging.info("*" * 50)
        logging.info("\n" + str(balance_check.tail(5)))
        logging.info("Positive: " + str(balance_check['positive'].sum()))
        logging.info("Negative: " + str(balance_check['negative'].sum()))
        logging.info("=" * 50)
        logging.info("=" * 50)
