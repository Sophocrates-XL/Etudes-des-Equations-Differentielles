import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import discrete as disc
import preprocessor as prep


# Verification 1: Validation of discrete model for rectangular space with Dirichlet boundary of zero potential,
# for which the solutions for heat equation and Poisson equation are well-known.
val_n_row = 48
val_n_col = 48
val_cell_matrix = np.zeros(shape = (val_n_row, val_n_col))
val_f_matrix = np.zeros(shape = (val_n_row, val_n_col))
for i in range(1, val_n_row - 1):
    for j in range(1, val_n_col - 1):
        val_cell_matrix[i,j] = prep.Layout.FLOOR
        val_f_matrix[i,j] = 1

val_layout = prep.Layout(val_cell_matrix, val_f_matrix)
val_graph_info = val_layout.toGraphInfo()

val_A = val_graph_info.A
val_B = val_graph_info.B
val_f = val_graph_info.f
val_floor_nodes = val_graph_info.floor_nodes
val_floor_A = val_A[np.ix_(val_floor_nodes, val_floor_nodes)]
val_floor_B = val_B[np.ix_(val_floor_nodes, val_floor_nodes)]
val_floor_f = val_f[val_floor_nodes]
val_floor_equation = disc.HeatEquation(A = val_floor_A, B = val_floor_B, f = val_floor_f, diffusivity = 1)

t1 = 50
t2 = 100
t3 = 200
phi0 = np.zeros(shape = (len(val_floor_nodes),))
val_floor_t1_state = val_floor_equation.getStateAtTime(phi0, t1)
val_floor_t2_state = val_floor_equation.getStateAtTime(phi0, t2)
val_floor_t3_state = val_floor_equation.getStateAtTime(phi0, t3)
val_floor_steady_state = val_floor_equation.getSteadyState()

val_t1_state = np.zeros(shape = (val_n_row * val_n_col,))
val_t2_state = np.zeros(shape = (val_n_row * val_n_col,))
val_t3_state = np.zeros(shape = (val_n_row * val_n_col,))
val_steady_state = np.zeros(shape = (val_n_row * val_n_col,))
for i in range(0, len(val_floor_nodes)):
    val_t1_state[val_floor_nodes[i]] = val_floor_t1_state[i]
    val_t2_state[val_floor_nodes[i]] = val_floor_t2_state[i]
    val_t3_state[val_floor_nodes[i]] = val_floor_t3_state[i]
    val_steady_state[val_floor_nodes[i]] = val_floor_steady_state[i]

val_t1_matrix = np.reshape(val_t1_state, newshape = (val_n_row, val_n_col))
val_t2_matrix = np.reshape(val_t2_state, newshape = (val_n_row, val_n_col))
val_t3_matrix = np.reshape(val_t3_state, newshape = (val_n_row, val_n_col))
val_steady_matrix = np.reshape(val_steady_state, newshape = (val_n_row, val_n_col))

val_steady_max = np.max(val_steady_matrix)
plt.imshow(val_t1_matrix, vmin = 0, vmax = val_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_t2_matrix, vmin = 0, vmax = val_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_t3_matrix, vmin = 0, vmax = val_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_steady_matrix, vmin = 0, vmax = val_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()


# Verification 2: Validation of discrete model for rectangular space with Dirichlet and Neumann boundary,
# for which the solutions for heat equation and Poisson equation are well-known.
val_2_n_row = 48
val_2_n_col = 48
val_2_cell_matrix = np.zeros(shape = (val_2_n_row, val_2_n_col))
val_2_f_matrix = np.zeros(shape = (val_2_n_row, val_2_n_col))
for j in range(0, val_2_n_col):
    val_2_cell_matrix[0,j] = prep.Layout.WALL
for i in range(1, val_2_n_row - 1):
    for j in range(1, val_2_n_col - 1):
        val_2_cell_matrix[i,j] = prep.Layout.FLOOR
        val_2_f_matrix[i,j] = 1

val_2_layout = prep.Layout(val_2_cell_matrix, val_2_f_matrix)
val_2_graph_info = val_2_layout.toGraphInfo()

val_2_A = val_2_graph_info.A
val_2_B = val_2_graph_info.B
val_2_f = val_2_graph_info.f
val_2_floor_nodes = val_2_graph_info.floor_nodes
val_2_floor_A = val_2_A[np.ix_(val_2_floor_nodes, val_2_floor_nodes)]
val_2_floor_B = val_2_B[np.ix_(val_2_floor_nodes, val_2_floor_nodes)]
val_2_floor_f = val_2_f[val_2_floor_nodes]
val_2_floor_equation = disc.HeatEquation(A = val_2_floor_A, B = val_2_floor_B, f = val_2_floor_f, diffusivity = 1)

t1 = 70
t2 = 140
t3 = 280
phi0 = np.zeros(shape = (len(val_2_floor_nodes),))
val_2_floor_t1_state = val_2_floor_equation.getStateAtTime(phi0, t1)
val_2_floor_t2_state = val_2_floor_equation.getStateAtTime(phi0, t2)
val_2_floor_t3_state = val_2_floor_equation.getStateAtTime(phi0, t3)
val_2_floor_steady_state = val_2_floor_equation.getSteadyState()

val_2_t1_state = np.zeros(shape = (val_2_n_row * val_2_n_col,))
val_2_t2_state = np.zeros(shape = (val_2_n_row * val_2_n_col,))
val_2_t3_state = np.zeros(shape = (val_2_n_row * val_2_n_col,))
val_2_steady_state = np.zeros(shape = (val_2_n_row * val_2_n_col,))
for i in range(0, len(val_2_floor_nodes)):
    val_2_t1_state[val_2_floor_nodes[i]] = val_2_floor_t1_state[i]
    val_2_t2_state[val_2_floor_nodes[i]] = val_2_floor_t2_state[i]
    val_2_t3_state[val_2_floor_nodes[i]] = val_2_floor_t3_state[i]
    val_2_steady_state[val_2_floor_nodes[i]] = val_2_floor_steady_state[i]

val_2_t1_matrix = np.reshape(val_2_t1_state, newshape = (val_2_n_row, val_2_n_col))
val_2_t2_matrix = np.reshape(val_2_t2_state, newshape = (val_2_n_row, val_2_n_col))
val_2_t3_matrix = np.reshape(val_2_t3_state, newshape = (val_2_n_row, val_2_n_col))
val_2_steady_matrix = np.reshape(val_2_steady_state, newshape = (val_2_n_row, val_2_n_col))

val_2_steady_max = np.max(val_2_steady_matrix)
plt.imshow(val_2_t1_matrix, vmin = 0, vmax = val_2_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_2_t2_matrix, vmin = 0, vmax = val_2_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_2_t3_matrix, vmin = 0, vmax = val_2_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()
plt.imshow(val_2_steady_matrix, vmin = 0, vmax = val_2_steady_max, cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()


# Case study 1.
n_row = 22
n_col = 34

# Prepares the layout matrix specifying cell types.
church_cell_matrix = np.zeros(shape = (n_row, n_col))
for i in range(5, 17):
    for j in range(1, 33):
        church_cell_matrix[i,j] = prep.Layout.FLOOR
for i in range(1, 21):
    for j in range(20, 26):
        church_cell_matrix[i,j] = prep.Layout.FLOOR
for i in [4, 17]:
    for j in [0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 26, 27, 32, 33]:
        church_cell_matrix[i,j] = prep.Layout.WALL
for i in [0, 21]:
    for j in [19, 20, 25, 26]:
        church_cell_matrix[i,j] = prep.Layout.WALL
for i in [0, 1, 2, 3, 4, 17, 18, 19, 20, 21]:
    for j in [19, 26]:
        church_cell_matrix[i,j] = prep.Layout.WALL

# Prepares the matrix of f.
church_f_matrix: np.ndarray = np.zeros(shape = (n_row, n_col))
for i in [6, 7, 8, 9, 12, 13, 14, 15]:
    for j in range(5, 20):
        church_f_matrix[i,j] = 3
for i in range(7, 15):
    for j in range(22, 26):
        church_f_matrix[i,j] = 1
for i in range(6, 16):
    for j in range(28, 32):
        church_f_matrix[i,j] = 1

# Prepares the adjacency matrix.
church_layout = prep.Layout(church_cell_matrix, church_f_matrix).magnify(3)
church_graph_info = church_layout.toGraphInfo()

church_A = church_graph_info.A
church_B = church_graph_info.B
church_f = church_graph_info.f
church_floor_nodes = church_graph_info.floor_nodes
church_floor_A = church_A[np.ix_(church_floor_nodes, church_floor_nodes)]
church_floor_B = church_B[np.ix_(church_floor_nodes, church_floor_nodes)]
church_floor_f = church_f[church_floor_nodes]
church_floor_equation = disc.HeatEquation(A = church_floor_A, B = church_floor_B, f = church_floor_f, diffusivity = 1)
church_floor_steady_state = church_floor_equation.getSteadyState()
church_steady_state = np.zeros(shape = (n_row * n_col * 3 * 3,))
for i in range(0, len(church_floor_nodes)):
    church_steady_state[church_floor_nodes[i]] = church_floor_steady_state[i]
church_steady_matrix = np.reshape(church_steady_state, newshape = (n_row * 3, n_col * 3))

plt.imshow(church_steady_matrix, cmap = "viridis", interpolation = "nearest")
plt.show()



# Case study 2.
market_cell_matrix = np.array(pd.read_csv(".\\Discrete\\Market-cell.csv", header = None).dropna(axis = 1))
market_f_matrix = np.array(pd.read_csv(".\\Discrete\\Market-f.csv", header = None).dropna(axis = 1))
market_layout = prep.Layout(market_cell_matrix, market_f_matrix).magnify(3)
market_n_row = market_layout.n_row
market_n_col = market_layout.n_col
market_graph_info = market_layout.toGraphInfo()

market_A = market_graph_info.A
market_B = market_graph_info.B
market_f = market_graph_info.f
market_floor_nodes = market_graph_info.floor_nodes

market_floor_A = market_A[np.ix_(market_floor_nodes, market_floor_nodes)]
market_floor_B = market_B[np.ix_(market_floor_nodes, market_floor_nodes)]
market_floor_f = market_f[market_floor_nodes]
market_floor_equation = disc.HeatEquation(A = market_floor_A, B = market_floor_B, f = market_floor_f, diffusivity = 1)

market_floor_steady_state = market_floor_equation.getSteadyState()
market_steady_state = np.zeros(shape = (market_n_row * market_n_col,))
for i in range(0, len(market_floor_nodes)):
    market_steady_state[market_floor_nodes[i]] = market_floor_steady_state[i]
market_steady_matrix = np.reshape(market_steady_state, newshape = (market_n_row, market_n_col))
market_steady_max = np.max(market_steady_matrix)

plt.imshow(market_steady_matrix, vmin = 0, vmax = market_steady_max,
    cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()


# Case study 3.
market_2_cell_matrix = np.array(pd.read_csv(".\\Discrete\\market_2-cell.csv", header = None).dropna(axis = 1))
market_2_f_matrix = np.array(pd.read_csv(".\\Discrete\\market_2-f.csv", header = None).dropna(axis = 1))
market_2_layout = prep.Layout(market_2_cell_matrix, market_2_f_matrix).magnify(3)
market_2_n_row = market_2_layout.n_row
market_2_n_col = market_2_layout.n_col
market_2_graph_info = market_2_layout.toGraphInfo()
market_2_A = market_2_graph_info.A
market_2_B = market_2_graph_info.B
market_2_f = market_2_graph_info.f
market_2_floor_nodes = market_2_graph_info.floor_nodes

market_2_floor_A = market_2_A[np.ix_(market_2_floor_nodes, market_2_floor_nodes)]
market_2_floor_B = market_2_B[np.ix_(market_2_floor_nodes, market_2_floor_nodes)]
market_2_floor_f = market_2_f[market_2_floor_nodes]
market_2_floor_equation = disc.HeatEquation(A = market_2_floor_A, B = market_2_floor_B, f = market_2_floor_f, diffusivity = 1)

market_2_floor_steady_state = market_2_floor_equation.getSteadyState()
market_2_steady_state = np.zeros(shape = (market_2_n_row * market_2_n_col,))
for i in range(0, len(market_2_floor_nodes)):
    market_2_steady_state[market_2_floor_nodes[i]] = market_2_floor_steady_state[i]
market_2_steady_matrix = np.reshape(market_2_steady_state, newshape = (market_2_n_row, market_2_n_col))
market_2_steady_max = np.max(market_2_steady_matrix)

plt.imshow(market_2_steady_matrix, vmin = 0, vmax = market_steady_max,
    cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()


# Case study 4.
market_3_cell_matrix = np.array(pd.read_csv(".\\Discrete\\market_3-cell.csv", header = None).dropna(axis = 1))
market_3_f_matrix = np.array(pd.read_csv(".\\Discrete\\market_3-f.csv", header = None).dropna(axis = 1))
market_3_layout = prep.Layout(market_3_cell_matrix, market_3_f_matrix).magnify(3)
market_3_n_row = market_3_layout.n_row
market_3_n_col = market_3_layout.n_col
market_3_graph_info = market_3_layout.toGraphInfo()

market_3_A = market_3_graph_info.A
market_3_B = market_3_graph_info.B
market_3_f = market_3_graph_info.f
market_3_floor_nodes = market_3_graph_info.floor_nodes

market_3_floor_A = market_3_A[np.ix_(market_3_floor_nodes, market_3_floor_nodes)]
market_3_floor_B = market_3_B[np.ix_(market_3_floor_nodes, market_3_floor_nodes)]
market_3_floor_f = market_3_f[market_3_floor_nodes]
market_3_floor_equation = disc.HeatEquation(A = market_3_floor_A, B = market_3_floor_B, f = market_3_floor_f, diffusivity = 1)

market_3_floor_steady_state = market_3_floor_equation.getSteadyState()
market_3_steady_state = np.zeros(shape = (market_3_n_row * market_3_n_col,))
for i in range(0, len(market_3_floor_nodes)):
    market_3_steady_state[market_3_floor_nodes[i]] = market_3_floor_steady_state[i]
market_3_steady_matrix = np.reshape(market_3_steady_state, newshape = (market_3_n_row, market_3_n_col))
market_3_steady_max = np.max(market_3_steady_matrix)

plt.imshow(market_3_steady_matrix, vmin = 0, vmax = market_steady_max,
    cmap = "viridis", interpolation = "nearest")
plt.show()
plt.clf()