import numpy as np


def get_matrix_by_list_index(matrix_base, list_index):
    temp_base_vector = None
    for index in list_index:
        if temp_base_vector is None:
            temp_base_vector = matrix_base[:index]
        else:
            temp_base_vector = np.column_stack(temp_base_vector, matrix_base[:index])
    return temp_base_vector
