from src.data_mining import DataMining


if __name__ == '__main__':
    # Create Dataset
    print("1.\tCriando os conjuntos de dados")
    DataMining.create()
    # Load Dataset
    print("2.\tCarregando os dados limpos")
    SONGSET = DataMining.load_song_set()
    SONGSET.info(memory_usage='deep')
    print("\n")
    USERSET = DataMining.load_user_set()
    USERSET.info(memory_usage='deep')
    print("2.\tPre Processamento")
    # DATASET = preprocessing.main_start(SONGSET)
    # DATASET.info(memory_usage='deep')
    # print("\n")
