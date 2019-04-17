from .data_mining import DataMining
from .textual_clean import TextualClean


class Preprocessing:
    @staticmethod
    def preprocessing():
        # Create Dataset
        print("*" * 50)
        print("1.1.\tCriando os conjuntos de dados")
        print("-" * 50)
        song_set, user_set = DataMining.create()
        print("*" * 50)
        # Load Dataset
        print("*" * 50)
        print("1.2.\tCarregando os dados ligados")
        print("-" * 50)
        # song_set = DataMining.load_song_set()
        song_set.info(memory_usage='deep')
        print("\n\n\n")
        # user_set = DataMining.load_user_set()
        user_set.info(memory_usage='deep')
        print("*" * 50)
        print("1.3.\tLimpando o texto")
        print("-" * 50)
        songs_df = TextualClean.main_start(song_set)
        songs_df.info(memory_usage='deep')
        DataMining.update_songs(songs_df)
        print("\n")
        return songs_df, user_set

    @staticmethod
    def load_data():
        return DataMining.load_song_set(), DataMining.load_user_set()

    @staticmethod
    def load_data_test():
        return DataMining.load_set_test()
