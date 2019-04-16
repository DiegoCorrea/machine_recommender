import multiprocessing


class GlobalVariable:
    processor_number = multiprocessing.cpu_count() - 1
    execution_times = 3
    user_min_song_list = 3

