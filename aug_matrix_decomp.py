import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt

from generate_matrices import get_coords

class Decomposer(nn.Module):
    def __init__(self, n_farms, n_crops, n_dims):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n_farms, n_dims, requires_grad=True).to(device))
        self.U = nn.Parameter(torch.randn(n_crops, n_dims, requires_grad=True).to(device))

    def forward(self):
        return self.V.mm(self.U.T).to(device)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

if __name__ == "__main__":
    #######################################
    ###   DEFINE EXECUTION PARAMETERS   ###
    #######################################

    # Define matrix name
    filename = 'matrix_20_5_20_0.25.csv'

    # Define matrix size
    n_grid = 20
    max_dims = 3
    n_crops = 20

    # Define training hyperparms
    epochs = 5000
    low = 0.1
    high = 0.9
    n_regimes = 3
    dist_lambda = 1

    # Set manual seeds
    torch.manual_seed(42)
    np.random.seed(42)

    #######################################
    #######################################

    model_dir = os.path.join('aug_models_results', filename[:-4]+f'{dist_lambda}')
    os.mkdir(model_dir)

    filepath = os.path.join('data', filename)
    matrix = pd.read_csv(filepath, index_col=0)
    matrix = torch.Tensor(matrix.values)

    coords = get_coords(n_grid)
    distances = pairwise_distances(coords)

    n_farms = n_grid * n_grid

    df_optim_d = {'pct_train': [], 'optim_d': [], 'test_optim_d': [], 'val_optim_d': []}

    for pct_train in np.linspace(low, high, n_regimes):

        print(f'Using {pct_train*100}% of the data for training')

        pct_test = (1 - pct_train) / 2
        pct_val = 1 - pct_train - pct_test

        all_train_loss_hist = []
        all_test_loss_hist = []
        all_val_loss_hist = []

        n_elements = n_farms * n_crops
        train_sz = int(pct_train * n_elements)
        test_sz = int(pct_test * n_elements)
        val_sz = n_elements - train_sz - test_sz

        random_perm = torch.randperm(n_elements)

        train_mask = torch.zeros(matrix.shape).to(device)
        test_mask = torch.zeros(matrix.shape).to(device)
        val_mask = torch.zeros(matrix.shape).to(device)


        def get_ij(idx, shape):
            _, n_columns = shape
            i = idx // n_columns
            j = idx % n_columns
            return i, j


        for idx in random_perm[:train_sz]:
            i, j = get_ij(idx, matrix.shape)
            train_mask[i][j] = 1

        for idx in random_perm[test_sz:train_sz + test_sz]:
            i, j = get_ij(idx, matrix.shape)
            test_mask[i][j] = 1

        for idx in random_perm[train_sz + test_sz:]:
            i, j = get_ij(idx, matrix.shape)
            val_mask[i][j] = 1

        for n_dims_model in range(1, max_dims+1):
            print('training with {} dimensions'.format(n_dims_model))
            decomposer = Decomposer(n_farms, n_crops, n_dims_model)

            opt = optim.Adam(decomposer.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            train_loss_hist = []
            test_loss_hist = []
            val_loss_hist = []

            for epoch in range(epochs):
                opt.zero_grad()
                matrix_pred = decomposer()

                farms_pred = decomposer.V
                distances_pred = pairwise_distances(farms_pred)

                estimation_loss = loss_fn(matrix_pred * train_mask, matrix * train_mask) * (n_elements / train_sz)
                distance_loss = loss_fn(n_dims_model**(1/2) * distances_pred, 2**(1/2) * distances)

                loss = estimation_loss + dist_lambda * distance_loss
                loss.backward()
                opt.step()

                train_loss_hist.append(loss.detach().numpy())

                test_loss = loss_fn(matrix_pred * test_mask, matrix * test_mask) \
                            * (n_elements / test_sz)

                test_loss_hist.append(test_loss.detach().numpy())

                val_loss = loss_fn(matrix_pred * val_mask, matrix * val_mask) \
                           * (n_elements / val_sz)

                val_loss_hist.append(val_loss.detach().numpy())

            all_train_loss_hist.append(train_loss_hist)
            all_test_loss_hist.append(test_loss_hist)
            all_val_loss_hist.append(val_loss_hist)

        train_df = pd.DataFrame(np.array(all_train_loss_hist).T)
        test_df = pd.DataFrame(np.array(all_test_loss_hist).T)
        val_df = pd.DataFrame(np.array(all_val_loss_hist).T)

        train_df.to_csv(os.path.join(model_dir, f'{int(100*pct_train)}_pct_acces_train_loss.csv'))
        test_df.to_csv(os.path.join(model_dir, f'{int(100*pct_train)}_pct_acces_test_loss.csv'))
        val_df.to_csv(os.path.join(model_dir, f'{int(100*pct_train)}_pct_acces_val_loss.csv'))

        fig = plt.figure(figsize=(9, 6))
        plt.plot(test_df)
        plt.ylabel('Test Loss')
        plt.xlim((0, 5000))
        plt.xlabel('Epoch')
        plt.title(f'Loss Over Epochs with Access to {int(100*pct_train)}% of Data')
        plt.legend(["{} dimensions".format(n) for n in range(1, 11)])
        plt.savefig(os.path.join(model_dir, f'{int(100*pct_train)}_pct_acces_loss_fig.eps'), dpi=200, format='eps')

        fig = plt.figure(figsize=(9, 6))
        plt.bar(range(1,max_dims+1), test_df.values[-1], color='orange')
        plt.ylabel('Test Loss')
        plt.xlabel('Number of Dimensions')
        plt.title('Loss vs Complexity with Access to {int(100*pct_train)}% of Data')
        plt.xticks(range(1, max_dims+1))
        plt.savefig(os.path.join(model_dir, f'{int(100*pct_train)}_pct_acces_dim_loss.eps'), dpi=200, format='eps')

        optim_d = val_df.values[-1].argmin()
        print('Optimal Number of dimensions is', optim_d + 1)
        print('Validation Loss is', float(val_df.values[-1][optim_d]))

        df_optim_d['pct_train'].append(pct_train)
        df_optim_d['optim_d'].append(optim_d+1)
        df_optim_d['test_optim_d'].append(test_df.values[-1][optim_d])
        df_optim_d['val_optim_d'].append(val_df.values[-1][optim_d])

    df_optim_d = pd.DataFrame(df_optim_d)
    df_optim_d.to_csv(os.path.join(model_dir, 'summary.csv'))

    fig = plt.figure(figsize=(9, 6))
    plt.plot(df_optim_d['val_optim_d'])
    plt.ylabel('Validation Loss')
    plt.xlabel('Percent of Visible Data for Training')
    plt.title(f'Validation Loss vs Percentage of Visible Data')
    plt.xticks(range(0,n_regimes), df_optim_d['pct_train'])
    plt.savefig(os.path.join(model_dir, f'summary_val_loss.eps'), dpi=200, format='eps')

    print('Done!')