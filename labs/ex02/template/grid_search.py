# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""
import numpy
import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
def grid_search(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.

    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """

    losses = np.zeros((len(grid_w0), len(grid_w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    w = np.hstack((grid_w0[:, np.newaxis], grid_w1[:, np.newaxis]))
    for i in range(len(grid_w0)):
        for j in range(len(grid_w1)):
            w = np.array([grid_w0[i], grid_w1[j]])
            losses[i, j] = compute_loss(y, tx, w)

    # ***************************************************
    return losses


# ***************************************************

def grid_search_v2(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ).
        grid_w1: numpy array of shape=(num_grid_pts_w1, ).

    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """
    row, col = np.indices((len(grid_w0), len(grid_w1)))
    big_w0, big_w1 = grid_w0[row.ravel()], grid_w1[col.ravel()]
    w = np.column_stack((big_w0, big_w1))  # num_grid_pts_w0*num_grid_pts_w1 x 2

    # N x 1 - N x 2 @ 2 x num_grid_pts_w0*num_grid_pts_w1 => N x num_grid_pts_w0*num_grid_pts_w1
    e = (y[:, np.newaxis] - tx @ w.T) ** 2
    nb_sample = y.shape[0]
    losses = 1 / (2 * nb_sample) * np.sum(e, axis=0)  # (num_grid_pts_w0*num_grid_pts_w1,)
    return losses.reshape((len(grid_w0), len(grid_w1)))


def grid_search_v3(y, tx, grid_w0, grid_w1):
    w0_size, w1_size = grid_w0.shape[0], grid_w1.shape[0]
    w0 = np.broadcast_to(grid_w0[:, np.newaxis], (w0_size, w1_size))
    w1 = np.broadcast_to(grid_w1[np.newaxis, :], (w0_size, w1_size))
    w = np.column_stack((w0.ravel(), w1.ravel()))
    e = (y[:, np.newaxis] - tx @ w.T) ** 2
    nb_sample = y.shape[0]
    losses = 1 / (2 * nb_sample) * np.sum(e, axis=0)
    return losses.reshape((len(grid_w0), len(grid_w1)))
