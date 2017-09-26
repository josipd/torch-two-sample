from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from itertools import combinations
from numpy.testing import assert_allclose
from torch_two_sample import (SmoothFRStatistic, SmoothKNNStatistic,
                              MMDStatistic)


# The code is based on the implementation [1] accompanying the following paper:
#
#  Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy
#  Sutherland et al.
#
# As we just use this to test, note that we do not square the bandwith in the
# denominiator.
#
# [1] https://github.com/dougalsutherland/opt-mmd
def shogun_mmd(X, Y, kernel_width, null_samples=1000, median_samples=1000,
               cache_size=32):
    '''
    Run an MMD test using a Gaussian kernel.

    Parameters
    ----------
    X : row-instance feature array

    Y : row-instance feature array

    kernel_width : float
        The bandwidth of the RBF kernel (sigma).

    null_samples : int
        How many times to sample from the null distribution.

    Returns
    -------
    p_val : float
        The obtained p value of the test.

    stat : float
        The test statistic.

    null_samples : array of length null_samples
        The samples from the null distribution.
    '''
    import modshogun as sg
    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(X.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(Y.T.astype(np.float64)))
    mmd.set_kernel(sg.GaussianKernel(cache_size, float(kernel_width)))

    mmd.set_num_null_samples(null_samples)
    samps = mmd.sample_null()
    stat = mmd.compute_statistic()

    p_val = np.mean(stat <= samps)
    return p_val, stat, samps


def test_mean_std():
    np.random.seed(0)
    for _ in range(10):
        n_1 = np.random.randint(3, 8)
        n_2 = np.random.randint(3, 8)
        n = n_1 + n_2
        dim = np.random.randint(3, 30)
        k = np.random.randint(1, min(n_1, n_2))
        alpha = 0.1 + 4 * np.random.rand()

        x = np.asarray(np.random.randn(n, dim))
        x[:n_1, :] += np.random.rand()

        samples_fr = []
        samples_knn = []

        loss_fn_fr = SmoothFRStatistic(n_1, n_2, False, compute_t_stat=True)
        loss_fn_knn = SmoothKNNStatistic(n_1, n_2, False, k,
                                         compute_t_stat=True)
        for set_1 in combinations(list(range(n)), n_1):
            set_2 = [i for i in range(n) if i not in set_1]
            assert len(set_2) == n_2
            assert len(set_1) == n_1
            var_1 = Variable(torch.from_numpy(x[set_1, :])).float()
            var_2 = Variable(torch.from_numpy(x[set_2, :])).float()
            samples_fr.append(loss_fn_fr(var_1, var_2, alphas=[alpha]).data[0])
            samples_knn.append(
                loss_fn_knn(var_1, var_2, alphas=[alpha]).data[0])

        def check(x_1, x_2):
            assert_allclose(x_1, x_2, atol=1e-2, rtol=1e-2)

        check(np.mean(samples_knn), 0)
        check(np.mean(samples_fr), 0)
        check(np.std(samples_knn), 1)
        check(np.std(samples_fr), 1)


def test_mmd():
    dim = 2
    torch.manual_seed(0)
    for n in range(30, 35):
        for fact in (0.2, 0.5, 0.8):
            n_1 = int(fact * n)
            n_2 = n - n_1
            sample_1 = torch.randn(n_1, dim)
            sample_2 = torch.randn(n_2, dim)
            sample_2[0, :] += 1.
            bandwith = 2
            shogun_pval, shogun_stat = shogun_mmd(
                sample_1.numpy(), sample_2.numpy(), kernel_width=bandwith,
                null_samples=100000)[:2]

            mmd_loss = MMDStatistic(n_1, n_2)
            stat, mat = mmd_loss(sample_1, sample_2, alphas=[1. / bandwith],
                                 ret_matrix=True)
            pval = mmd_loss.pval(mat, n_permutations=100000)

            # Shogun normalizes the statistic, so we normalize ours.
            assert np.isclose(stat * (n_1 * n_2) / (n_1 + n_2), shogun_stat,
                              rtol=1e-3)
            assert np.abs(pval - shogun_pval) < 0.01
