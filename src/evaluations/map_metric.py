# -*- coding: utf-8 -*-
from src.globalVariable import GlobalVariable


class MAPController:
    @staticmethod
    def get_ap_from_list(relevance_array):
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        hit_list = []
        relevant = 0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                relevant += 1
            hit_list.append(relevant / (i + 1))
        ap = sum(hit_list)
        if ap > 0.0:
            return ap / relevance_list_size
        else:
            return 0.0

    @staticmethod
    def at_all_position(relevance_array):
        return [MAPController.get_ap_from_list(relevance_array[:at]) for at in GlobalVariable.AT_SIZE_LIST]
