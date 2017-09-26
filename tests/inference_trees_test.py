from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from torch_two_sample.inference_trees import TreeMarginals


def test_cardinality():
    """Check that the sum of all edges is always n-1."""
    torch.manual_seed(0)
    for n in range(5, 51):
        fn = TreeMarginals(n, False)
        unaries = torch.randn(1, n * (n - 1) // 2)
        margs = fn(Variable(unaries))
        assert np.isclose(margs.sum().data[0], n - 1)


def test_chain():
    """Test that on a chain graph the edges present in the chain have marginals
       close to one and the other have marginals close to zero."""
    for n in range(3, 51):
        fn = TreeMarginals(n, False)

        unaries = -10000 * torch.ones(n * (n - 1) // 2)
        k = 0
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1:
                    unaries[k] = 1
                    indices.append(1)
                else:
                    indices.append(0)
                k += 1

        margs = fn(Variable(unaries)).cpu().data.numpy()
        indices = np.asarray(indices, dtype=np.bool)
        assert np.allclose(margs[indices], 1)
        assert np.allclose(margs[~indices], 0)


def test_cycle():
    """Test that on a cycle graph the edges present in the cycle have marginals
       close to n / (n - 1) and the other have marginals close to zero."""
    for n in range(3, 51):
        fn = TreeMarginals(n, False)

        unaries = -10000 * torch.ones(n * (n - 1) // 2)
        k = 0
        indices = []
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or i == 0 and j + 1 == n:
                    unaries[k] = 1
                    indices.append(1)
                else:
                    indices.append(0)
                k += 1

        margs = fn(Variable(unaries)).cpu().data.numpy()
        indices = np.asarray(indices, dtype=np.bool)
        assert np.allclose(margs[indices], (n - 1) / n)
        assert np.allclose(margs[~indices], 0)


def test_dumbbell():
    """Test that on the dumbbell graph two complete graphs connected using a
       single edge, the single edge gets a probability of one and the ones
       in the complete graphs get uniform probabilities."""
    for m in range(3, 51):
        n = 2 * m
        fn = TreeMarginals(n, False)

        unaries = -10000 * torch.ones(n * (n - 1) // 2)
        k = 0
        indices = np.zeros(n * (n - 1) // 2, dtype=np.bool)
        for i in range(n):
            for j in range(i + 1, n):
                if i < m and j < m or i >= m and j >= m:
                    unaries[k] = 1
                    indices[k] = 1
                elif i + 1 == m == j:
                    unaries[k] = 1
                    dumbbell_idx = k
                k += 1

        margs = fn(Variable(unaries)).cpu().data.numpy()
        indices = np.asarray(indices, dtype=np.bool)
        assert np.allclose(margs[indices], (m - 1) / (m * (m - 1) // 2),
                           rtol=1e-3)
        assert np.allclose(margs[dumbbell_idx], 1., rtol=1e-3)


def test_triu():
    """Check that conversion from and to to the triangular form works."""
    fn = TreeMarginals(4, False)
    matrix = torch.Tensor([
        [1,  2,  3, 4],
        [5,  6,  7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]])
    triu = fn.triu(Variable(matrix))
    assert np.allclose(triu.data.numpy(), np.array([
        2, 3, 4, 7, 8, 12]))
    matrix_recovered = fn.to_mat(triu).data
    assert np.allclose(matrix_recovered.numpy(), np.array([
        [0, 2, 3, 4],
        [0, 0, 7, 8],
        [0, 0, 0, 12],
        [0, 0, 0, 0]]))
