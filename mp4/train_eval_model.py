"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    x = data["image"]
    y = data["label"]
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
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)
    total_grad = model.backward(f, y_batch)
    model.w = model.w - learning_rate * total_grad


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    pass
    # Set model.w
    feature_size = model.ndims + 1
    model.w = z[0:feature_size]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    x = data["image"]
    y = data["label"]
    all_ones = np.ones(x.shape[0])
    x = np.c_[x, all_ones]
    sample_size = x.shape[0]
    feature_size = model.ndims + 1

    P0 = np.eye(feature_size)
    P1 = np.zeros((sample_size, feature_size))
    P2 = np.zeros((sample_size+feature_size, sample_size))
    P3 = np.r_[P0, P1]
    P = model.w_decay_factor * np.c_[P3, P2]

    q0 = np.zeros(feature_size)
    q1 = np.ones(sample_size)
    q =  np.r_[q0, q1]
    q = q.reshape((q.shape[0], 1))

    G0 = -y * x
    G1 = -np.eye(sample_size)
    G2 = np.c_[G0, G1]
    G3 = np.zeros((sample_size, feature_size))
    G4 = np.eye(sample_size)
    G5 = -np.c_[G3, G4]
    G = np.r_[G2, G5]

    h0 = -np.ones((sample_size, 1))
    h1 = np.zeros((sample_size, 1))
    h = np.r_[h0, h1]

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    x = data["image"]
    y = data["label"]
    all_ones = np.ones(x.shape[0])
    x = np.c_[x, all_ones]

    f = np.dot(x, model.w)
    loss = model.total_loss(f, y)

    y_predict = model.predict(f)
    acc = np.mean(y_predict == y) * 100
    return loss, acc
