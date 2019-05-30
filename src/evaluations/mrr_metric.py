# -*- coding: utf-8 -*-
from src.globalVariable import GlobalVariable


class MRRController:
    @staticmethod
    def get_rr_from_list(relevance_array):
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                return 1 / (i + 1)
        return 0.0

    @staticmethod
    def at_all_position(relevance_array):
        return [MRRController.get_rr_from_list(relevance_array[:at]) for at in GlobalVariable.AT_SIZE_LIST]
