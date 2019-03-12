from src.preprocessing_clean import PreProcessing

pre = PreProcessing()
result = pre.link_songs()
print(result.head())
