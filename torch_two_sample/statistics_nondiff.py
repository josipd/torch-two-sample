"""The classical non-differentiable Friedman-Rafsky and k-NN tests."""

from scipy.sparse.csgraph import minimum_spanning_tree as mst
from torch.autograd import Function
import numpy as np
import torch
from .permutation_test import permutation_test_mat
from .util import pdist

__all__ = ['FRStatistic', 'KNNStatistic']


class MSTFn(Function):
    """Compute the minimum spanning tree given a matrix of pairwise weights."""
    def forward(self, weights):
        """Compute the MST given the edge weights.

        The behaviour is the same as that of ``minimum_spanning_tree` in
        ``scipy.sparse.csgraph``, namely i) the edges are assumed non-negative,
        ii) if ``weights[i, j]`` and ``weights[j, i]`` are both non-negative,
        their minimum is taken as the edge weight.

        Arguments
        ---------
        weights: :class:`torch:torch.Tensor`
            The adjacency matrix of size ``(n, n)``.

        Returns
        -------
        :class:`torch:torch.Tensor`
            An ``(n, n)`` matrix adjacency matrix of the minimum spanning tree.

            Indices corresponding to the edges in the MST are set to one, rest
            are set to zero.

            If both weights[i, j] and weights[j, i] are non-zero, then the one
            will be located in whichever holds the *smaller* value (ties broken
            arbitrarily).
        """
        mst_matrix = mst(weights.cpu().numpy()).toarray() > 0
        assert int(mst_matrix.sum()) + 1 == weights.size(0)
        return torch.Tensor(mst_matrix.astype(float))


class KSmallest(Function):
    """Return an indicator vector holing the smallest k elements in each row.

    Arguments
    ---------
    k: int
        How many elements to keep per row."""
    def __init__(self, k):
        super(KSmallest, self).__init__()
        self.k = k

    def forward(self, matrix):
        """Compute the positions holding the largest k elements in each row.

        Arguments
        ---------
        matrix: :class:`torch:torch.Tensor`
            Tensor of size (n, m)

        Returns
        -------
        torch.Tensor of size (n, m)
           The positions that correspond to the k largest elements are set to
           one, the rest are set to zero."""
        self.mark_non_differentiable(matrix)
        matrix = matrix.numpy()
        indices = np.argsort(matrix, axis=1)
        mins = np.zeros_like(matrix)
        rows = np.arange(matrix.shape[0]).reshape(-1, 1)
        mins[rows, indices[:, :self.k]] = 1
        return torch.Tensor(mins)


class FRStatistic(object):
    """The classical Friedman-Rafsky test :cite:`friedman1979multivariate`.

    Arguments
    ----------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample."""
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed Friedman-Rafsky test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.FRStatistic.pval`.

        Returns
        -------
        float
            The number of edges that do connect points from the *same* sample.
        """
        n_1 = sample_1.size(0)
        assert n_1 == self.n_1 and sample_2.size(0) == self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        mst_matrix = MSTFn()(diffs)

        statistic = mst_matrix[:n_1, :n_1].sum() + mst_matrix[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, mst_matrix
        else:
            return statistic

    def pval(self, mst, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.FRStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(mst.data.numpy(),
                                    self.n_1, self.n_2, n_permutations)


class KNNStatistic(object):
    """The classical k-NN test :cite:`friedman1983graph`.

    Arguments
    ---------
    n_1: int
        The number of data points in the first sample.
    n_2: int
        The number of data points in the second sample
    k: int
        The number of nearest neighbours (k in kNN).
    """
    def __init__(self, n_1, n_2, k):
        self.n_1 = n_1
        self.n_2 = n_2
        self.k = k

    def __call__(self, sample_1, sample_2, norm=2, ret_matrix=False):
        """Evaluate the non-smoothed kNN test statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, variable of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, variable of size ``(n_1, d)``.
        norm: float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.KNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The number of edges that connect points from the *same* sample.
        :class:`torch:torch.autograd.Variable` (optional)
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1 and n_2 == self.n_2
        n = self.n_1 + self.n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)

        indices = (1. - torch.eye(n)).byte()
        if sample_12.is_cuda:
            indices = indices.cuda()

        for i in range(n):
            diffs[i, i] = float('inf')  # We don't want the diagonal selected.
        smallest = KSmallest(self.k)(diffs.cpu())
        statistic = smallest[:n_1, :n_1].sum() + smallest[n_1:, n_1:].sum()

        if ret_matrix:
            return statistic, smallest
        else:
            return statistic

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.KNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)
