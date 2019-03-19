from src.data_mining import DataMining


if __name__ == '__main__':
    # Create Dataset
    print("*"*50)
    print("1.\tCriando os conjuntos de dados")
    print("-" * 50)
    # DataMining.create()
    print("*" * 50)
    # Load Dataset
    print("*" * 50)
    print("2.\tCarregando os dados limpos")
    print("-" * 50)
    SONGSET = DataMining.load_song_set()
    SONGSET.info(memory_usage='deep')
    print(str(SONGSET.song_id.nunique()))
    print("\n")
    USERSET = DataMining.load_user_set()
    USERSET.info(memory_usage='deep')
    print(str(USERSET.user_id.nunique()))
    print("*" * 50)
    print("3.\tPre Processamento")
    print("-" * 50)
    # DATASET = preprocessing.main_start(SONGSET)
    # DATASET.info(memory_usage='deep')
    # print("\n")
