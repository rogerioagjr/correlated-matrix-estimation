import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_coords(n_grid):
    n_farms = n_grid * n_grid
    coords = []
    for farm in range(n_farms):
        coords.append([farm % n_grid, farm // n_grid])
    coords = torch.Tensor(coords)
    coords = coords / n_grid - 0.5
    return coords

def save_matrix_csv(matrix, n_grid, n_dim, n_crops, sigma):
    filename = 'matrix_{}_{}_{}_{}.csv'.format(n_grid, n_dim, n_crops, sigma)
    df = pd.DataFrame(matrix)
    fileptath = os.path.join('data', filename)
    df.to_csv(fileptath)

def save_matrix_fig(matrix, n_grid, n_dim, n_crops, sigma):
    n_farms = n_grid * n_grid
    filename = 'matrix_{}_{}_{}_{}.eps'.format(n_grid, n_dim, n_crops, sigma)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(matrix, interpolation='nearest', aspect=n_crops / n_farms)
    fileptath = os.path.join('images', filename)
    plt.savefig(fileptath, dpi=200, format='eps')


if __name__ == "__main__":
    # Define matrix size
    n_grid = 20
    n_dims = 5
    n_crops = 20
    # Define matrix noise
    sigma = 0.25
    # Set manual seeds
    torch.manual_seed(42)
    np.random.seed(42)

    coords = get_coords(n_grid)

    encoder = 2 * torch.rand((2, n_dims)) - 1
    farms = torch.mm(coords, encoder)
    crops = 2 * torch.rand((n_crops, n_dims)) - 1

    raw_matrix = torch.mm(farms, crops.T)
    norm_matrix = (raw_matrix-raw_matrix.mean())/raw_matrix.std()
    noise = torch.normal(0, sigma, raw_matrix.shape)

    matrix = raw_matrix + noise
    matrix = matrix.numpy()

    save_matrix_csv(matrix, n_grid, n_dims, n_crops, sigma)
    save_matrix_fig(matrix, n_grid, n_dims, n_crops, sigma)