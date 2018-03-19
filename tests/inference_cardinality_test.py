from __future__ import division, print_function

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch_two_sample.inference_cardinality import inference_cardinality, NINF


def test_softmax():
    torch.manual_seed(0)
    for batch_size in range(100, 123):
        for k in range(2, 10):
            unaries = Variable(torch.randn(batch_size, 15))
            count_potentials = Variable(NINF * torch.ones(batch_size, k + 1))
            count_potentials[:, 1] = 0
            output_softmax = softmax(unaries, dim=1)
            output_cardinf = inference_cardinality(unaries, count_potentials)
            assert np.allclose(output_softmax.cpu().data.numpy(),
                               output_cardinf.cpu().data.numpy())


def test_uniform():
    torch.manual_seed(0)
    np.random.seed(0)
    for batch_size in range(100, 123):
        for k in range(2, 10):
            unaries = Variable(torch.randn(batch_size, 1)).expand(
                batch_size, 15)
            count_potentials = Variable(NINF * torch.ones(batch_size, k + 1))
            z = np.random.randint(1, k + 1)
            count_potentials[:, z] = 1
            output_cardinf = inference_cardinality(unaries, count_potentials)
            assert np.allclose(output_cardinf.sum(1).cpu().data.numpy(), z,
                               rtol=1e-3)
