"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    x = processed_dataset[0]
    y = processed_dataset[1]
    sample_size = x.shape[0]
    batch_num = int(np.ceil(sample_size / batch_size))
    batch_tuple = [(i * batch_size, min(sample_size, (i + 1) * batch_size))
                   for i in range(0, batch_num)]
    index_array = np.arange(sample_size)
    for step in range(num_steps):
        if shuffle == True:
            np.random.shuffle(index_array)
        for batch_index, (batch_start, batch_end) in enumerate(batch_tuple):
            batch_id = index_array[batch_start:batch_end]
            update_step(x[batch_id], y[batch_id], model, learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    total_grad = model.backward(f, y_batch)
    model.w = model.w - learning_rate * total_grad




def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]

    one_array = np.ones(x.shape[0])
    x_new = np.c_[x, one_array].astype(int)

    a = np.dot(x_new.T, x_new) + model.w_decay_factor * np.eye(x_new.shape[1])
    b = np.dot(x_new.T, y)
    model.w = np.dot(np.linalg.inv(a), b)


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]

    one_array = np.ones(x.shape[0])
    x_new = np.c_[x, one_array].astype(int)

    f = np.dot(x_new, model.w)
    loss = model.total_loss(f, y)

    return loss
