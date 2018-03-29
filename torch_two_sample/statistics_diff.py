"""Differentiable two-sample tests."""
from __future__ import division, print_function

import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

from .inference_cardinality import inference_cardinality, NINF
from .permutation_test import permutation_test_tri, permutation_test_mat
from .inference_trees import TreeMarginals
from .util import pdist


class SmoothFRStatistic(object):
    r"""The smoothed Friedman-Rafsky test :cite:`djolonga17graphtests`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample.
    cuda: bool
        If true, the arguments to :py:meth:`~.SmoothFRStatistic.__call__` must
        be be on the current cuda device. Otherwise, they should be on the cpu.
    """
    def __init__(self, n_1, n_2, cuda, compute_t_stat=True):
        n = n_1 + n_2
        self.n_1, self.n_2 = n_1, n_2
        # The idx_within tensor contains the indices that correspond to edges
        # that connect samples from within the same sample.
        # The matrix self.nbs is of size (n, n_edges) and has 1 in position
        # (i, j) if node i is incident to edge j. Specifically, note that
        # self.nbs @ mu will result in a vector that has at position i the sum
        # of the marginals of all edges incident to i, which we need in the
        # formula for the variance.
        if cuda:
            self.idx_within = torch.cuda.ByteTensor((n * (n - 1)) // 2)
            if compute_t_stat:
                self.nbs = torch.cuda.FloatTensor(n, self.idx_within.size()[0])
        else:
            self.idx_within = torch.ByteTensor((n * (n - 1)) // 2)
            if compute_t_stat:
                self.nbs = torch.FloatTensor(n, self.idx_within.size()[0])
        self.idx_within.zero_()
        if compute_t_stat:
            self.nbs.zero_()
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if compute_t_stat:
                    self.nbs[i, k] = 1
                    self.nbs[j, k] = 1
                if (i < n_1 and j < n_1) or (i >= n_1 and j >= n_1):
                    self.idx_within[k] = 1
                k += 1

        self.marginals_fn = TreeMarginals(n_1 + n_2, cuda)
        self.compute_t_stat = compute_t_stat

    def __call__(self, sample_1, sample_2, alphas, norm=2, ret_matrix=False):
        r"""Evaluate the smoothed Friedman-Rafsky test statistic.

        The test accepts several **inverse temperatures** in ``alphas``, does
        one test for each ``alpha``, and takes their mean as the statistic.
        Namely, using the notation in :cite:`djolonga17graphtests`, the
        value returned by this call if ``compute_t_stat=False`` is equal to:

        .. math::

            -\frac{1}{m}\sum_{j=m}^k T_{\pi^*}^{1/\alpha_j}(\textrm{sample}_1,
                                                            \textrm{sample}_2).

        If ``compute_t_stat=True``, the returned value is the t-statistic of
        the above quantity under the permutation null. Note that we compute the
        negated statistic of what is used in :cite:`djolonga17graphtests`, as
        it is exactly what we want to minimize when used as an objective for
        training implicit models.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, should be of size ``(n_1, d)``.
        sample_2: :class:`torch:torch.autograd.Variable`
            The second sample, should be of size ``(n_2, d)``.
        alphas: list of :class:`float` numbers
            The inverse temperatures.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.SmoothFRStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic, a t-statistic if ``compute_t_stat=True``.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12, norm=norm)
        margs = None
        for alpha in alphas:
            margs_a = self.marginals_fn(
                self.marginals_fn.triu(- alpha * diffs))
            if margs is None:
                margs = margs_a
            else:
                margs = margs + margs_a

        margs = margs / len(alphas)
        idx_within = Variable(self.idx_within, requires_grad=False)
        n_1, n_2, n = self.n_1, self.n_2, self.n_1 + self.n_2
        m = margs.sum()
        t_stat = m - torch.masked_select(margs, idx_within).sum()
        if self.compute_t_stat:
            nbs = Variable(self.nbs, requires_grad=False)
            nbs_sum = (nbs.mm(margs.unsqueeze(1))**2).sum()
            chi_1 = n_1 * n_2 / (n * (n - 1))
            chi_2 = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))
            var = (chi_1 * (1 - chi_2) * nbs_sum +
                   chi_1 * chi_2 * (margs**2).sum() +
                   chi_1 * (chi_2 - 4 * chi_1) * m**2)
            mean = 2 * m * n_1 * n_2 / (n * (n - 1))
            std = torch.sqrt(1e-5 + var)
        else:
            mean = 0.
            std = 1.

        if ret_matrix:
            return - (t_stat - mean) / std, margs
        else:
            return - (t_stat - mean) / std

    def pval(self, matrix, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.SmoothFRStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_tri(matrix.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)


class SmoothKNNStatistic(object):
    r"""The smoothed k-nearest neighbours test :cite:`djolonga17graphtests`.

    Note that the ``k=1`` case is computed directly using a SoftMax and should
    execute much faster than the statistics with ``k > 1``.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample.
    cuda: bool
        If true, the arguments to ``__call__`` must be be on the current
        cuda device. Otherwise, they should be on the cpu.
    k: int
        The number of nearest neighbours (k in kNN)."""
    def __init__(self, n_1, n_2, cuda, k, compute_t_stat=True):
        self.count_potential = torch.FloatTensor(1, k + 1)
        self.count_potential.fill_(NINF)
        self.count_potential[0, -1] = 0
        self.indices_cpu = (1 - torch.eye(n_1 + n_2)).byte()
        self.k = k
        self.n_1 = n_1
        self.n_2 = n_2
        self.cuda = cuda
        if cuda:
            self.indices = self.indices_cpu.cuda()
        else:
            self.indices = self.indices_cpu
        self.compute_t_stat = compute_t_stat

    def __call__(self, sample_1, sample_2, alphas, norm=2, ret_matrix=False):
        r"""Evaluate the smoothed kNN statistic.

        The test accepts several **inverse temperatures** in ``alphas``, does
        one test for each ``alpha``, and takes their mean as the statistic.
        Namely, using the notation in :cite:`djolonga17graphtests`, the
        value returned by this call if `compute_t_stat=False` is equal to:

        .. math::

            -\frac{1}{m}\sum_{j=m}^k T_{\pi^*}^{1/\alpha_j}(\textrm{sample}_1,
                                                            \textrm{sample}_2).

        If ``compute_t_stat=True``, the returned value is the t-statistic of
        the above quantity under the permutation null. Note that we compute the
        negated statistic of what is used in :cite:`djolonga17graphtests`, as
        it is exactly what we want to minimize when used as an objective for
        training implicit models.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alpha: list of :class:`float`
            The smoothing strengths.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.SmoothKNNStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic, a t-statistic if ``compute_t_stat=True``.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        n_1 = sample_1.size(0)
        n_2 = sample_2.size(0)
        assert n_1 == self.n_1
        assert n_2 == self.n_2
        n = n_1 + n_2
        sample_12 = torch.cat((sample_1, sample_2), 0)
        diffs = pdist(sample_12, sample_12)
        indices = Variable(self.indices, requires_grad=False)
        indices_cpu = Variable(self.indices_cpu, requires_grad=False)
        k = self.count_potential.size()[1] - 1
        assert k == self.k
        count_potential = Variable(
            self.count_potential.expand(n, k + 1), requires_grad=False)

        diffs = torch.masked_select(diffs, indices).view(n, n - 1)

        margs_ = None
        for alpha in alphas:
            if self.k == 1:
                margs_a = softmax(-alpha * diffs, dim=1)
            else:
                margs_a = inference_cardinality(
                    - alpha * diffs.cpu(), count_potential)
            if margs_ is None:
                margs_ = margs_a
            else:
                margs_ = margs_ + margs_a

        margs_ /= len(alphas)
        # The variable margs_ is a matrix of size n x n-1, which we want to
        # reshape to n x n by adding a zero diagonal, as it makes the following
        # logic easier to follow. The variable margs_ is on the GPU when k=1.
        if margs_.is_cuda:
            margs = torch.cuda.FloatTensor(n, n)
        else:
            margs = torch.FloatTensor(n, n)
        margs.zero_()
        margs = Variable(margs, requires_grad=False)
        if margs_.is_cuda:
            margs.masked_scatter_(indices, margs_.view(-1))
        else:
            margs.masked_scatter_(indices_cpu, margs_.view(-1))

        t_stat = margs[:n_1, n_1:].sum() + margs[n_1:, :n_1].sum()
        if self.compute_t_stat:
            m = margs.sum()
            mean = 2 * m * n_1 * n_2 / (n * (n - 1))
            nbs_sum = ((
                margs.sum(0).view(-1) + margs.sum(1).view(-1))**2).sum()
            flip_sum = (margs * margs.transpose(1, 0)).sum()
            chi_1 = n_1 * n_2 / (n * (n - 1))
            chi_2 = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))
            var = (chi_1 * (1 - chi_2) * nbs_sum +
                   chi_1 * chi_2 * (margs**2).sum() +
                   chi_1 * chi_2 * flip_sum +
                   chi_1 * (chi_2 - 4 * chi_1) * m ** 2)
            std = torch.sqrt(1e-5 + var)
        else:
            mean = 0.
            std = 1.

        if ret_matrix:
            return - (t_stat - mean) / std, margs
        else:
            return - (t_stat - mean) / std

    def pval(self, margs, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.SmoothKNNStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(margs.data.cpu().numpy(),
                                    self.n_1, self.n_2, n_permutations)


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.

    The kernel used is equal to:

    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},

    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.

        The kernel used is

        .. math::

            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},

        for the provided ``alphas``.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        if isinstance(distances, Variable):
            distances = distances.data
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)


class EnergyStatistic:
    r"""The energy test of :cite:`szekely2013energy`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = - 1. / (n_1 * n_1)
        self.a11 = - 1. / (n_2 * n_2)
        self.a01 = 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, ret_matrix=False):
        r"""Evaluate the statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.EnergyStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        d_1 = distances[:self.n_1, :self.n_1].sum()
        d_2 = distances[-self.n_2:, -self.n_2:].sum()
        d_12 = distances[:self.n_1, -self.n_2:].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        if ret_matrix:
            return loss, distances
        else:
            return loss

    def pval(self, distances, n_permutations=1000):
        """Compute a p-value using a permutation test.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.EnergyStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.

        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(distances.data.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)
