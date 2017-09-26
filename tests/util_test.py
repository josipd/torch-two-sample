from __future__ import division, print_function

import numpy as np
import torch
from torch_two_sample.util import pdist
from scipy.spatial.distance import squareform, pdist as scipy_pdist


def test_pdist():
    torch.manual_seed(0)
    np.random.seed(1)
    for n_1 in range(10, 100, 10):
        for n_2 in range(10, 100, 10):
            dim = np.random.randint(1, 10)
            sample_1 = torch.randn(n_1, dim)
            sample_2 = torch.randn(n_2, dim)
            p = 1 + 2 * np.random.rand()  # Use this l_p norm.
            distances = pdist(sample_1, sample_2, norm=p, eps=1e-9).numpy()
            sample_12 = np.vstack((sample_1.numpy(), sample_2.numpy()))
            distances_scipy = squareform(scipy_pdist(
                sample_12, metric='minkowski', p=p))
            print(distances - distances_scipy[:n_1, n_1:])
            assert np.allclose(distances, distances_scipy[:n_1, n_1:],
                               rtol=1e-4, atol=1e-4)
