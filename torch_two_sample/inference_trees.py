r"""Perform marginals inference in models of the form

  p(x) = exp(\sum_i z_ix_i) nu(x) / Z,

where nu(x) is one if x forms a valid spanning tree, or zero otherwise."""
from __future__ import division, print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import relu


class TreeMarginals(object):
    r"""Perform marginal inference in models over spanning trees.

    The model considered is of the form:

    .. math::
        p(x) \propto \exp(\sum_{i=1}^m d_i x_i) \nu(x),

    where :math:`x` is a binary random vector with one coordinate per edge,
    and :math:`\nu(x)` is one if :math:`x` forms a spanning tree, or zero
    otherwise.

    The numbers :math:`d_i` are expected to be given by taking the upper
    triangular part of the adjacecny matrix. To extract the upper triangular
    part of a matrix, or to reconstruct them matrix from it, you can use the
    functions :py:meth:`~.triu` and :py:meth:`~.to_mat`.

    Arguments
    ---------
    n_vertices: int
      The number of vertices in the graph.
    cuda: bool
      Should the function work on cuda (on the current device) or cpu."""
    def __init__(self, n_vertices, cuda):
        self.n_vertices = n_vertices

        self.triu_mask = torch.triu(
            torch.ones(n_vertices, n_vertices), 1).byte()
        if cuda:
            self.triu_mask = self.triu_mask.cuda()

        n_edges = n_vertices * (n_vertices - 1) // 2
        # A is the edge incidence matrix, arbitrarily oriented.
        if cuda:
            A = torch.cuda.FloatTensor(n_vertices, n_edges)
        else:
            A = torch.FloatTensor(n_vertices, n_edges)
        A.zero_()

        k = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                A[i, k] = +1
                A[j, k] = -1
                k += 1
        self.A = A[1:, :]  # We remove the first node from the matrix.

    def to_mat(self, triu):
        r"""Given the upper triangular part, reconstruct the matrix.

        Arguments
        ---------
        x: :class:`torch:torch.autograd.Variable`
            The upper triangular part, should be of size ``n * (n - 1) / 2``.

        Returns
        --------
        :class:`torch:torch.autograd.Variable`
          The ``(n, n)``-matrix whose upper triangular part filled in with
          ``x``, and the rest with zeroes"""
        if triu.is_cuda:
            matrix = torch.cuda.FloatTensor(self.n_vertices, self.n_vertices)
        else:
            matrix = torch.zeros(self.n_vertices, self.n_vertices)
        matrix.zero_()
        triu_mask = Variable(self.triu_mask, requires_grad=False)
        matrix = Variable(matrix, requires_grad=False)
        return matrix.masked_scatter(triu_mask, triu)

    def triu(self, matrix):
        r"""Given a matrix, extract its upper triangular part.

        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            A square matrix of size ``(n, n)``.

        Returns
        --------
        :class:`torch:torch.autograd.Variable`
          The upper triangular part of the given matrix, which is of size
          ``n * (n - 1) // 2``"""
        triu_mask = Variable(self.triu_mask, requires_grad=False)
        return torch.masked_select(matrix, triu_mask)

    def __call__(self, d):
        r"""Compute the marginals in the model.

        Arguments
        ---------
        d: :class:`torch:torch.autograd.Variable`
            A vector of size ``n * (n - 1) // 2`` containing the :math:`d_i`.

        Returns
        --------
        :class:`torch:torch.autograd.Variable`
            The marginal probabilities in a vector of size
            ``n * (n - 1) // 2``."""
        d = d - d.max()  # So that we don't have to compute large exponentials.

        # Construct the Laplacian.
        L_off = self.to_mat(torch.exp(d))
        L_off = L_off + L_off.t()
        L_dia = torch.diag(L_off.sum(1))
        L = L_dia - L_off
        L = L[1:, 1:]

        A = Variable(self.A, requires_grad=False)
        P = (1. / torch.diag(L)).view(1, -1)  # The diagonal pre-conditioner.
        Z, _ = torch.gesv(A, L * P.expand_as(L))
        Z = Z * P.t().expand_as(Z)
        # relu for numerical stability, the inside term should never be zero.
        return relu(torch.sum(Z * A, 0)) * torch.exp(d)
