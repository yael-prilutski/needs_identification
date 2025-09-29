import pickle
import itertools
import numpy as np
from os import listdir
from os.path import join
import matplotlib.pyplot as plt


COLOR_MAP = {-1: "#5E4FA2", 1: "#48A0B2",
             2: "#A1D9A4", 3: "#EDF7A3",
             4: "#FEE899", 5: "#FBA45C",
             6: "#E25249", 7: "#9E0142"}

DIM_AMP = {2: [1, 2], 3: [1, 3], 4: [3, 2], 5: [5, 2], 6: [5, 3], 7: [7, 3], 8: [7, 4], 9: [9, 4], 10: [6, 5]}


def _get_color_variable(variable, color_map, default):
    keys = np.array([x for x in color_map.keys()])
    if default:
        keys = keys - np.min(keys)
        corrected_var = variable - min(variable)
    else:
        corrected_var = np.array(variable)

    final_color_map = np.repeat("#606060", np.max(keys) + 1)
    values = np.array([color_map[x] for x in color_map.keys()])

    final_color_map[keys] = values

    return (final_color_map[corrected_var])


def visualize_dataset(mat, save_path, variable=None, color_map=None, default=True, alpha=0.3):
    if variable is not None and color_map is not None:
        color = _get_color_variable(variable, color_map, default)
    else:
        color = "#606060"

    dim = mat.shape[0]

    num_rows = DIM_AMP[dim][0]
    num_cols = DIM_AMP[dim][1]

    all_planes = [x for x in itertools.combinations(range(0, dim), 2)]

    # Create a figure and axis objects
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 15))

    # Loop through each subplot and set some properties
    plane_counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            dim1 = all_planes[plane_counter][0]
            dim2 = all_planes[plane_counter][1]
            plane_counter += 1

            x = mat[dim1, :]
            y = mat[dim2, :]

            ax = axs[i, j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.scatter(x, y, alpha=alpha, color=color, rasterized=True)
            ax.set_xlabel('Dim %d' % dim1)
            ax.set_ylabel('Dim %d' % dim2)

            # x_min, x_max = np.percentile(x, [10, 90])
            # y_min, y_max = np.percentile(y, [10, 90])
            # pad_x = (x_max - x_min) * 0.1
            # pad_y = (y_max - y_min) * 0.1
            # ax.set_xlim(x_min - pad_x, x_max + pad_x)
            # ax.set_ylim(y_min - pad_y, y_max + pad_y)

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_specific_dimension(mat, save_path, dim1_v, dim2_v, variable=None, color_map=None, alpha=0.3):
    color = _get_color_variable(variable, color_map, None)
    dim = mat.shape[0]

    num_rows = DIM_AMP[dim][0]
    num_cols = DIM_AMP[dim][1]

    all_planes = [x for x in itertools.combinations(range(0, dim), 2)]

    # Loop through each subplot and set some properties
    plane_counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            dim1 = all_planes[plane_counter][0]
            dim2 = all_planes[plane_counter][1]
            plane_counter += 1

            if dim1 == dim1_v and dim2 == dim2_v:
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.scatter(mat[dim1, :], mat[dim2, :], alpha=alpha, color=color, s=100)
                ax.set_xlabel('Dim %d' % dim1)
                ax.set_ylabel('Dim %d' % dim2)

                # plt.show()
                plt.savefig(save_path)
                plt.close()


def get_cluster_labels(data_name, dir_path):
    cluster_map_path = join(
        join(dir_path, data_name, 'cluster_map'), listdir(join(dir_path, data_name, 'cluster_map'))[0])
    with open(cluster_map_path, 'rb') as f:
        cluster_map = pickle.load(f)
    cluster_labels = cluster_map["cluster_labels"]

    return cluster_labels
