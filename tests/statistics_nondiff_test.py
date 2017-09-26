from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from torch_two_sample.statistics_nondiff import (KSmallest, MSTFn, FRStatistic,
                                                 KNNStatistic)
from torch_two_sample.statistics_diff import SmoothKNNStatistic


def test_k_smallest():
    smallest = KSmallest(2)(Variable(torch.Tensor([
        [1, 2, 3, 4, 5, 6],
        [2, 4, 1, 4, 5, 7],
        [-1, 1, 1, -4, 9, 3]]))).data
    assert np.allclose(smallest.numpy(), np.asarray([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0]]))

    smallest = KSmallest(1)(Variable(torch.Tensor([
        [1, 2, 3, 4, 5, 6],
        [2, 4, 1, 4, 5, 7],
        [-1, 1, 1, -4, 9, 3]]))).data
    assert np.allclose(smallest.numpy(), np.asarray([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]]))

    # Test a tie.
    smallest = KSmallest(2)(Variable(torch.Tensor([
        [1, 2, 2, 4, 5, 6],
        [2, 4, 1, 4, 5, 7],
        [-1, 1, 1, -1, 9, 3]]))).data
    option_1 = np.allclose(smallest.numpy(), np.asarray([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0]]))
    option_2 = np.allclose(smallest.numpy(), np.asarray([
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0]]))
    assert option_1 or option_2


def test_mst():
    # The following example is from the scipy documentation on mst.
    mst_matrix = Variable(torch.Tensor([
        [0, 8, 0, 3],
        [0, 0, 2, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 0]]))
    assert np.allclose(MSTFn()(mst_matrix).data.numpy(), np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]))


def test_fr():
    # We will test the one-dimensional case, in which case the MST simply
    # connects the points in their non-descending order.
    sample_1 = Variable(torch.Tensor([[5], [4], [2], [3], [1]]))
    sample_2 = Variable(torch.Tensor([[2.5], [-1], [-2], [6]]))
    # The ordering of between the samples is
    #   2 - 2 ~ 1 - 1 ~ 2 ~ 1 - 1 - 1 ~ 2
    # The total number of edges between the same class (marked -) is 4.
    fr_test = FRStatistic(5, 4)
    assert fr_test(sample_1, sample_2, ret_matrix=False).data[0] == 4.
    fr_test = FRStatistic(4, 5)
    assert fr_test(sample_2, sample_1, ret_matrix=False).data[0] == 4.


def test_1nn():
    # We will test the one-dimensional case.
    sample_1 = Variable(torch.Tensor([[5], [2], [4.1], [3], [1.1]]))
    sample_2 = Variable(torch.Tensor([[2.6], [-1.5], [-2], [6]]))
    #  sample:     2     2      1      1     2      1    1      1     2
    # ordered:    -2    -1.5    1.1    2     2.6    3    4.1    5     6
    #                ->             <-    <~     <~          ->
    #                <-                          ~>          <-    <~
    # The total number of edges between the same class (marked -) is 5.
    nn_test = KNNStatistic(5, 4, k=1)
    assert nn_test(sample_1, sample_2, ret_matrix=False).data[0] == 5.
    nn_test = KNNStatistic(4, 5, k=1)
    assert nn_test(sample_2, sample_1, ret_matrix=False).data[0] == 5.


def test_1nn_smooth():
    torch.manual_seed(0)
    for i in range(6, 100):
        n_1 = 15
        sample_1 = Variable(torch.randn(n_1, 4)) + 0.01
        n_2 = 10
        sample_2 = Variable(torch.randn(n_2, 4))

        nn_test = KNNStatistic(n_1, n_2, k=1)
        nn_test_smooth = SmoothKNNStatistic(
            n_1, n_2, False, k=1, compute_t_stat=False)
        stat = nn_test(sample_1, sample_2).data[0]
        stat_smooth = nn_test_smooth(sample_1, sample_2, [1e9]).data[0]
        # The smooth tests computes the *negative* of the number of edges that
        # connect points from *different* samples, so that n + that should
        # agree with the non-smooth test.
        assert np.isclose(stat, n_1 + n_2 + stat_smooth, rtol=1e-3)
