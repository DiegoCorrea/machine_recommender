from sklearn.model_selection import train_test_split

TEST_SIZE = 20


class ContentBased:
    @staticmethod
    def split_user_data(song_id_list, data_class):
        """
            Função particionadora dos dados em treino e teste
            As constantes estão no arquivo de variaveis de sistema
            :param data_df: Dados atributo/valor
            :param data_class: Classe dos dados
            :return: Quatro valores: dados de treino, classe de treino, dados de teste, classe de teste
            """
        return train_test_split(song_id_list, data_class, test_size=TEST_SIZE)

    @staticmethod
    def labeled_user_line(user_df):
        user_play_std = user_df['play_count'].std()
        user_df['like'] = ""
        for index, row in user_df.iterrows():
            user_df.at[index, 'like'] = True if row['play_count'] >= user_play_std else False
        return user_df

    @staticmethod
    def recommender():
        pass
