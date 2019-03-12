from src.preprocessing_clean import PreProcessing

pre = PreProcessing()
pre.link_songs()
pre.link_user_and_songs()
pre.save()
print((pre.get_song_df()).head())
