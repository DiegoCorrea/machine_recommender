class BaseAnalytics:
    @staticmethod
    def song_info(sonf_df):
        print("| INDEX |Quantidade de músicas: " + str(len(sonf_df.index)))
        print("| SONG_ID | Quantidade de músicas únicas: " + str(sonf_df['song_id'].nunique()))
        print("| TITLE | Quantidade de títulos únicos: " + str(sonf_df['title'].nunique()))
        print("| ALBUM | Quantidade de albuns únicas: " + str(sonf_df['album'].nunique()))
        print("| ARTIST | Quantidade de artista únicas: " + str(sonf_df['artist'].nunique()))
        print("| YEAR | Quantidade de anos únicas: " + str(sonf_df['year'].nunique()))
        print("| YEAR = 0 | Quantidade de anos zerados: " + str(len(sonf_df[sonf_df['year'] == '0'])))
        print(str(sonf_df['year'].unique()))

    @staticmethod
    def song_complete_info(sonf_df):
        BaseAnalytics.song_info(sonf_df)
        print("Quantidade de gênero únicas: " + str(sonf_df['genre'].nunique()))

    @staticmethod
    def raw_genre_info(genre_df):
        print("Quantidade de música com gênero: " + str(len(genre_df.index)))
        print("Quantidade de gêneros: " + str(genre_df['genre'].nunique()))

    @staticmethod
    def user_info(users_preferences_df):
        print("Quantidade de preferências: " + str(len(users_preferences_df.index)))
        print("Únicos usuários: " + str(users_preferences_df['user_id'].nunique()))
        print("Músicas ouvidas usuários: " + str(users_preferences_df['song_id'].nunique()))

    @staticmethod
    def complete_info(users_preferences_df):
        pass
