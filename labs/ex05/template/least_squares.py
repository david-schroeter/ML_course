# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx @ w
    return e.T @ e / (2 * len(y))


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w_star = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w_star, compute_mse(y, tx, w_star)
