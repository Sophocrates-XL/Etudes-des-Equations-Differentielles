from __future__ import annotations

import numpy as np

class Layout:

    ENVIRONMENT = 0
    WALL = 1
    FLOOR = 2

    def __init__(self, cell_matrix: np.ndarray, f_matrix: np.ndarray):

        self.cell_matrix = cell_matrix
        self.f_matrix = f_matrix
        self.n_row = cell_matrix.shape[0]
        self.n_col = cell_matrix.shape[1]

    def magnify(self, n: int) -> Layout:

        old_cell_matrix = self.cell_matrix
        old_f_matrix = self.f_matrix
        new_cell_matrix = np.ndarray(shape = (self.n_row * n, self.n_col * n))
        new_f_matrix = np.ndarray(shape = (self.n_row * n, self.n_col * n))
        f_divisor = n * n
        for i in range(0, self.n_row):
            for j in range(0, self.n_col):
                new_i = range(i * n, (i + 1) * n)
                new_j = range(j * n, (j + 1) * n)
                new_cell_matrix[np.ix_(new_i, new_j)] = old_cell_matrix[i,j]
                new_f_matrix[np.ix_(new_i, new_j)] = old_f_matrix[i,j] / f_divisor

        return Layout(new_cell_matrix, new_f_matrix)


    def toGraphInfo(self) -> GraphInfo:

        cell_matrix = self.cell_matrix

        n_row: int = cell_matrix.shape[0]
        n_col: int = cell_matrix.shape[1]

        A = np.zeros(shape=(n_row * n_col, n_row * n_col))
        B = np.zeros(shape=(n_row * n_col, n_row * n_col))
        f = np.reshape(self.f_matrix, newshape = (n_row * n_col,))

        floor_nodes = []
        for i in range(1, n_row - 1):
            for j in range(1, n_col - 1):
                if cell_matrix[i, j] == Layout.FLOOR:
                    i_node = i * n_col + j
                    floor_nodes.append(i_node)
                    if cell_matrix[i - 1, j] == Layout.FLOOR:
                        A[i_node, i_node - n_col] = 1
                        A[i_node - n_col, i_node] = 1
                    elif cell_matrix[i - 1, j] == Layout.ENVIRONMENT:
                        B[i_node, i_node] += 1
                    if cell_matrix[i + 1, j] == Layout.FLOOR:
                        A[i_node, i_node + n_col] = 1
                        A[i_node + n_col, i_node] = 1
                    elif cell_matrix[i + 1, j] == Layout.ENVIRONMENT:
                        B[i_node, i_node] += 1
                    if cell_matrix[i, j - 1] == Layout.FLOOR:
                        A[i_node, i_node - 1] = 1
                        A[i_node - 1, i_node] = 1
                    elif cell_matrix[i, j - 1] == Layout.ENVIRONMENT:
                        B[i_node, i_node] += 1
                    if cell_matrix[i, j + 1] == Layout.FLOOR:
                        A[i_node, i_node + 1] = 1
                        A[i_node + 1, i_node] = 1
                    elif cell_matrix[i, j + 1] == Layout.ENVIRONMENT:
                        B[i_node, i_node] += 1

        return GraphInfo(A = A, B = B, f = f, floor_nodes = floor_nodes)



class GraphInfo:

    def __init__(self, A, B, f, floor_nodes):
        self.A = np.copy(A)
        self.B = np.copy(B)
        self.f = np.copy(f)
        self.floor_nodes = list(floor_nodes)