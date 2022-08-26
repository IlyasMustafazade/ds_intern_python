import sys, os, logging
import numpy as np
import matplotlib.pyplot as plt

FILE_DEPTH = 3
sys.path.append(
    "\\".join(os.path.abspath(__file__).split("\\")[:-FILE_DEPTH]))
from modules.dl.layer import Input
from modules.dl.activ import identity


class NN:
    def __init__(self, in_dim=None, loss_fn=None, n_iter=None, optimizer=None):
        self.lyr_lst = [Input(in_dim)]
        self.output = None
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.history = None

    def add(self, attr_tpl_lst, n_neuron_lst=None):
        for i, attr_tpl in enumerate(attr_tpl_lst):
            lyr_type, activ_fn = attr_tpl
            self.lyr_lst.append(lyr_type(self.lyr_lst[-1].out_dim,
                n_neuron_lst[i], activ_fn))

    def forward(self, in_mtx, save=True):
        last_activ = in_mtx
        for lyr in self.lyr_lst:
            last_activ = lyr.forward(last_activ, save=save)
        if save is True:
            self.output = last_activ
        return last_activ

    def backward(self, lbl):
        last = self.lyr_lst[-1]
        activ_deriv = last.activ_fn(last.output, deriv=True)
        loss_deriv = self.loss_fn(output=last.activ, act=lbl, deriv=True)
        delta = loss_deriv * activ_deriv
        for middle, right in zip(self.lyr_lst[-2::-1], self.lyr_lst[::-1]):
            right.set_d_wgt(middle.activ @ delta.T)
            right.set_d_bias(np.sum(delta, axis=1, keepdims=True))
            activ_deriv = middle.activ_fn(middle.output, deriv=True)
            delta = (right.wgt @ delta) * activ_deriv

    def fit(self, train_ftr, test_ftr, train_lbl, test_lbl):
        self.history = History()
        for epoch in range(self.n_iter):
            self.forward(train_ftr)
            self.history.record("train", self.loss_fn(self.output, train_lbl))
            test_pred = self.forward(test_ftr, save=False)
            self.history.record("test", self.loss_fn(test_pred, test_lbl))
            self.backward(train_lbl)
            self.optimizer(self)


class History:
    def __init__(self):
        self.train_loss = np.empty((0,))
        self.test_loss = np.empty((0,))

    def record(self, loss_type, value):
        as_1d = np.squeeze(np.array([value]))
        if loss_type == "test":
            self.test_loss = np.append(self.test_loss, as_1d)
        elif loss_type == "train":
            self.train_loss = np.append(self.train_loss, as_1d)

    def plot(self):
        fig = plt.figure(num="Train loss and test loss vs epoch number")
        x = np.arange(len(self.train_loss))
        s = 0.5
        plt.scatter(x, self.train_loss, c="green", s=s)
        plt.scatter(x, self.test_loss, c="red", s=s)
        plt.show()
