# uses ptr_class for xlsx data type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ptr_class import Pointer_like


class LIFE_Matrix:
    def __init__(self, file_path):
        self.matrix_df, self.uber_df = self.read_matrix(file_path)
        self.edge_names = self.extract_edge_names()
        self.node_names = self.extract_node_names()
        self.num_nodes = len(self.node_names)
        self.num_edges = len(self.edge_names)

        self.x_array = self.generate_x_array()
        self.node_name_to_x_index = self.generate_node_name_to_index_dict()
        self.uber_edges = self.determine_uber_edges()
        self.s_matrix = self.generate_S_matrix()

    def read_matrix(self, filepath):
        matrix_df = pd.read_excel(filepath, sheet_name="Hyper edges")
        uber_df = pd.read_excel(filepath, sheet_name="Uber edges")
        return matrix_df, uber_df

    def extract_edge_names(self):
        columns = self.matrix_df.columns.values
        return columns[1:]  # we ignore the first entry - it's metabolite\reaction

    def extract_node_names(self):
        node_name_col = self.matrix_df["metabolite\\edge"]
        return node_name_col.values

    def generate_x_array(self):
        num_nodes = len(self.node_names)
        zero_ptr = Pointer_like(0)
        x_array_ptr = [zero_ptr for ele in range(0, num_nodes)]
        return x_array_ptr

    def generate_node_name_to_index_dict(self):
        node_name_dict = {}
        i = 0
        for name in self.node_names:
            node_name_dict[name] = i
            i += 1

        return node_name_dict

    def determine_uber_edges(self):
        uber_edge_names = self.uber_df.columns.values[
            1:
        ]  # again, first col is 'metabolite\reaction'
        uber_node_dict = {}
        uber_edge_dict = {}
        uber_edge_effect = {}
        for uber_edge in uber_edge_names:
            index_of_nonzero = self.uber_df[uber_edge].to_numpy().nonzero()[0]
            edges_affected = self.uber_df[uber_edge][index_of_nonzero].values[0]
            uber_edge_dict[uber_edge] = edges_affected
            if edges_affected[0] == "-":
                uber_edge_effect[uber_edge] = "negative"
                uber_edge_dict[uber_edge] = edges_affected[1:]
            else:
                uber_edge_effect[uber_edge] = "positive"
                uber_edge_dict[uber_edge] = edges_affected

            node_list = []
            for index in index_of_nonzero:
                node = self.node_names[index]
                node_list.append(node)
            uber_node_dict[uber_edge] = node_list

        edge_uber_dict = {}
        for key in uber_edge_dict.keys():
            val = uber_edge_dict[key]
            try:
                edge_uber_dict[val] += [key]
            except KeyError:
                edge_uber_dict[val] = [key]

        print(uber_node_dict)
        print(uber_edge_dict)
        print(edge_uber_dict)
        print(uber_edge_effect)

    def generate_S_matrix(self):
        # we'll just do some if, elifs to determine edge type by the name of the column, then generate the entry
        # zero_ptr = Pointer_like(0)
        for edge in self.edge_names:
            if edge[0] == "s":  # a simple edge
                continue
            elif edge[0] == "h":  # a hyper edge
                continue
            else:  # a sink or a source
                continue


def K_p(x):
    return Pointer_like(1)


def K_m(x):
    return Pointer_like(1)


def F(x):
    return x


def define_matrix_by_hand():
    x = [Pointer_like(1) for i in range(0, 20)]
    neg_one_ptr = Pointer_like(-1)
    r_0 = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[0] * K_p(x[6]),
        neg_one_ptr * x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_1 = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[1] * K_p(x[6]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_2 = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[2],
    ]
    r_3 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_4 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_5 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[5],
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_6 = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[6] * K_m(x[5]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_7 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[0] * K_p(x[6]),
        0.0,
        0.0,
        neg_one_ptr * x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
        0.0,
        neg_one_ptr * x[7],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_8 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[1] * K_p(x[6]),
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_9 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[6] * K_m(x[5]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_10 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[7],
        neg_one_ptr * x[10],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_11 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[10],
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[11],
        0.0,
        0.0,
        0.0,
    ]
    r_12 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
        x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_13 = [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[13] * K_m(x[11]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_14 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        x[13] * K_m(x[11]),
        neg_one_ptr * (x[14]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_15 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        F(x[14]),
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[15],
        0.0,
    ]
    r_16 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        F(x[14]),
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[16],
        0.0,
        0.0,
    ]
    r_17 = [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * F(x[17]) * K_p(x[16]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    r_18 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        F(x[17]) * K_p(x[16]),
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[18],
    ]
    r_19 = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        F(x[17]) * K_p(x[16]),
        0.0,
        0.0,
        0.0,
        0.0,
        neg_one_ptr * x[19],
    ]
    s_list = [
        r_0,
        r_1,
        r_2,
        r_3,
        r_4,
        r_5,
        r_6,
        r_7,
        r_8,
        r_9,
        r_10,
        r_11,
        r_12,
        r_13,
        r_14,
        r_15,
        r_16,
        r_17,
        r_18,
        r_19,
    ]
    for i in range(0, len(s_list)):
        for j in range(0, len(s_list[0])):
            if type(s_list[i][j]) != Pointer_like:
                s_list[i][j] = Pointer_like(s_list[i][j])
    s_matrix = np.array(s_list)
    return s_matrix


if __name__ == "__main__":
    matrix_file_name = "data/Updated PD Matrix.xlsx"
    # life_matrix = LIFE_Matrix(matrix_file_name)
    matrix_df = pd.read_excel(matrix_file_name, sheet_name="Hyper edges")
    node_names = matrix_df["metabolite\\edge"].values
    # temp = pd.read_excel('s_matrix_by_hand.xlsx', sheet_name='S_matrix')
    np.random.seed(0)
    vals = np.random.uniform(low=0.2, high=0.7, size=20)

    x = [Pointer_like(ele) for ele in np.random.rand(20)]
    # x = [Pointer_like(1), Pointer_like(0.5), Pointer_like(0.2), Pointer_like(0.4), Pointer_like(0.8), Pointer_like(0.2), Pointer_like(0.11), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1), Pointer_like(1)]
    zero_ptr = Pointer_like(0)
    one_ptr = Pointer_like(1)
    neg_one_ptr = Pointer_like(-1)
    # r_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, neg_one_ptr * x[0] * K_p(x[6]),
    #        neg_one_ptr * x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0, 0.0]
    # r_1 = [0.0, 1.0, 0.0, 0.0, 0.0, neg_one_ptr * x[1] * K_p(x[6]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0, 0.0, 0.0, 0.0]
    # r_2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, neg_one_ptr*x[2]]
    # r_3 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, neg_one_ptr*x[3]]
    # r_4 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, neg_one_ptr*x[4]]
    # r_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0, neg_one_ptr * x[5], 0.0, 0.0, 0.0, 0.0]
    # r_6 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, neg_one_ptr * x[6] * K_m(x[5]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0, 0.0, 0.0, 0.0]
    # r_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[0] * K_p(x[6]), 0.0, 0.0,
    #        neg_one_ptr * x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]), 0.0, neg_one_ptr * x[7], 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0, 0.0, 0.0]
    # r_8 = [0.0, 0.0, 0.0, 0.0, 0.0, x[1] * K_p(x[6]), 0.0, 0.0, 0.0, 0.0,
    #        neg_one_ptr * x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # r_9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[6] * K_m(x[5]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #        0.0, 0.0]
    # r_10 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[7], neg_one_ptr * x[10], 0.0, 0.0, 0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0]
    # r_11 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[10], 0.0, 0.0, 0.0, 0.0, neg_one_ptr * x[11],
    #         0.0, 0.0, 0.0]
    # r_12 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
    #         x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # r_13 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, neg_one_ptr * x[13] * K_m(x[11]), 0.0, 0.0,
    #         0.0, 0.0, 0.0, 0.0, 0.0]
    # r_14 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[13] * K_m(x[11]), neg_one_ptr * (x[14]),
    #         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # r_15 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, F(x[14]), 0.0, 0.0, 0.0, 0.0,
    #         neg_one_ptr * x[15], 0.0]
    # r_16 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, F(x[14]), 0.0, 0.0, 0.0,
    #         neg_one_ptr * x[16], 0.0, 0.0]
    # r_17 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #         neg_one_ptr * F(x[17]) * K_p(x[16]), 0.0, 0.0, 0.0, 0.0, 0.0]
    # r_18 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, F(x[17]) * K_p(x[16]), 0.0, 0.0,
    #         0.0, 0.0, neg_one_ptr * x[18]]
    # r_19 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, F(x[17]) * K_p(x[16]), 0.0, 0.0,
    #         0.0, 0.0, neg_one_ptr * x[19]]
    # s_list = [r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11, r_12, r_13, r_14, r_15, r_16, r_17, r_18,
    #           r_19]
    # for i in range(0, len(s_list)):
    #     for j in range(0, len(s_list[0])):
    #         if type(s_list[i][j]) != ptr_class.Pointer_like:
    #             # if s_list[i][j] == 1.0:
    #             #     s_list[i][j] = 0.4
    #             s_list[i][j] = Pointer_like(s_list[i][j])
    # s_matrix = np.array(s_list)

    stoich_matrix_ptr = np.array(
        [
            [
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[0] * K_p(x[6]),
                neg_one_ptr * x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[1] * K_p(x[6]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[2],
            ],
            [
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[3],
            ],
            [
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[4],
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[0] * K_p(x[2]) * K_p(x[3]) * K_p(x[4]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[5],
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[6] * K_m(x[5]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[0] * K_p(x[6]),
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
                zero_ptr,
                neg_one_ptr * x[7],
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[1] * K_p(x[6]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[6] * K_m(x[5]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[7],
                neg_one_ptr * x[10],
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[10],
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[11],
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[7] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
                x[8] * K_p(x[9]) * K_p(x[16]) * K_m(x[17]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[13] * K_m(x[11]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                x[13] * K_m(x[11]),
                neg_one_ptr * (x[14]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                F(x[14]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[15],
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                F(x[14]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[16],
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                one_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * F(x[17]) * K_p(x[16]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                F(x[17]) * K_p(x[16]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[18],
            ],
            [
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                F(x[17]) * K_p(x[16]),
                zero_ptr,
                zero_ptr,
                zero_ptr,
                zero_ptr,
                neg_one_ptr * x[19],
            ],
        ]
    )

    flux_ptr = np.array([Pointer_like(1)] * 21)

    t_current = 0
    t_step = 0.1
    t_final = 2
    t_array = [t_current]

    num_values = 10
    values = [[ele.get() for ele in x[0:num_values]]]

    t_step_ptr = Pointer_like(t_step)

    while t_current < t_final:
        # print("######################################")
        # print("######################################")
        # print("######################################")
        # print("######################################")
        # print('current_t:', t_current)
        # print('stoich_S:', stoich_matrix_ptr)
        # print('flux_ptr:', flux_ptr)
        # print('x_ptr:', x)
        integration_step = x + np.array(
            [ele * t_step_ptr for ele in np.matmul(stoich_matrix_ptr, flux_ptr)]
        )

        # update x:
        for i in range(0, len(integration_step)):
            x[i].set(integration_step[i].get())

        t_current += t_step
        t_array.append(t_current)
        values.append([ele.get() for ele in x[0:num_values]])

        # print('x_ptr values after update:', [ele.get() for ele in x])
        print("s_values:", [[item.get() for item in ele] for ele in stoich_matrix_ptr])

    plt.plot(t_array, values, label=node_names[0:num_values])
    plt.legend(loc="lower left")
    plt.show()
    # for i in range(0, temp.shape[0]):
    #     print('r_' + str(i) + ' = ' + str(temp.iloc[i, 1:].values.tolist()))
    # print(life_matrix.matrix_df)
    # print(life_matrix.uber_df)
