r"""Perform marginal inference in cardinality potentials.

These are models of the form:

.. math::

    p(x) = \exp( \sum_i x_iq_i + f(\sum_i x_i) ) / Z,

where :math:`x `is a Bernoulli random vector. The function :math:`f` is called
a cardinality potential, while the terms :math:`q_i` is known as the node
potentials.

Note that the cardinality and node potentials are *inside* the exponential.

The implemented methods are from the following paper:

  Cardinality Restricted Boltzmann Machines. Kevin Swersky et al., NIPS 2012,

and the code has been adapted from the numpy code accompanying it."""
from __future__ import division, print_function
import torch as torch
from torch.autograd import Variable

__all__ = 'inference_cardinality',

NINF = -1e+5  # TODO(josipd): Implement computation with negative infinities.


def logsumexp(x, dim):
    """Compute the log-sum-exp in a numerically stable way.

   Arguments
   ---------
   x : :class:`torch:torch.Tensor`
   dim : int
       The dimension along wich the operation should be computed.

   Returns
   --------
   :class:`torch:torch.Tensor`
       The dimension along which the sum is done is not squeezed.
    """
    x_max = torch.max(x, dim, keepdim=True)[0]
    return torch.log(torch.sum(
        torch.exp(x - x_max.expand_as(x)), dim, keepdim=True)) + x_max


def logaddexp(x, y):
    """Compute log(e^x + e^y) element-wise in a numerically stable way.

    The arguments have to be of equal dimension.

    Arguments
    ---------
    x : :class:`torch:torch.Tensor`
    y : :class:`torch:torch.Tensor`"""
    maxes = torch.max(x, y)
    return torch.log(torch.exp(x - maxes) + torch.exp(y - maxes)) + maxes


def compute_bwd(node_pot, msg_in):
    """Compute the new backward message from the given node potential and
    incoming message."""
    node_pot = node_pot.unsqueeze(1)
    msg_out = msg_in.clone()
    msg_out[:, 1:] = logaddexp(
        msg_out[:, 1:], node_pot.expand_as(msg_in[:, :-1]) + msg_in[:, :-1])
    # Normalize for numerical stability.
    return msg_out - logsumexp(msg_out, 1).expand_as(msg_out)


def compute_fwd(node_pot, msg_in):
    """Compute the new forward message from the given node potential and
    incoming message."""
    node_pot = node_pot.unsqueeze(1)
    msg_out = msg_in.clone()
    msg_out[:, :-1] = logaddexp(
        msg_out[:, :-1], node_pot.expand_as(msg_in[:, 1:]) + msg_in[:, 1:])
    # Normalize for numerical stability.
    return msg_out - logsumexp(msg_out, 1).expand_as(msg_out)


def inference_cardinality(node_potentials, cardinality_potential):
    r"""Perform inference in a graphical model of the form

    .. math::

        p(x) \propto \exp( \sum_{i=1}^n x_iq_i + f(\sum_{i=1}^n x_i) ),

    where :math:`x` is a binary random variable. The vector :math:`q` holds the
    node potentials, while :math:`f` is the so-called cardinality potential.

    Arguments
    ---------
    node_potentials: :class:`torch:torch.autograd.Variable`
        The matrix holding the per-node potentials :math:`q` of size
        ``(batch_size, n)``.
    cardinality_potentials: :class:`torch:torch.autograd.Variable`
        The cardinality potential.

        Should be of size ``(batch_size, n_potentials)``.
        In each row, column ``i`` holds the value :math:`f(i)`.
        If it happens ``n_potentials < n + 1``, the remaining positions are
        assumed to be equal to ``-inf`` (i.e., are given zero probability)."""
    def create_var(val, *dims):
        """Helper to initialize a variable on the right device."""
        if node_potentials.is_cuda:
            tensor = torch.cuda.FloatTensor(*dims)
        else:
            tensor = torch.FloatTensor(*dims)
        tensor.fill_(val)
        return Variable(tensor, requires_grad=False)

    batch_size, dim_node = node_potentials.size()
    assert batch_size == cardinality_potential.size()[0]

    fmsgs = []
    fmsgs.append(cardinality_potential.clone())
    for i in range(dim_node-1):
        fmsgs.append(compute_fwd(node_potentials[:, i], fmsgs[-1]))
    fmsgs.append(create_var(NINF, cardinality_potential.size()))

    bmsgs = []
    bmsgs.append(create_var(NINF, cardinality_potential.size()))
    bmsgs[0][:, 0] = 0
    bmsgs[0][:, 1] = node_potentials[:, dim_node-1]
    for i in reversed(range(1, dim_node)):
        bmsgs.insert(0, compute_bwd(node_potentials[:, i-1], bmsgs[0]))
    bmsgs.insert(0, create_var(NINF, cardinality_potential.size()))

    # Construct pairwise beliefs (without explicitly instantiating the D^2
    # size matrices), then sum the diagonal to get b0, and the off-diagonal
    # to get b1.  b1/(b0+b1) gives marginal for original y_d for all except
    # the last variable, y_D.  we need to special case it, because there is
    # no pairwise potential that represents \theta_D -- it's just a unary in
    # the transformed model.
    fmsgs = torch.cat([fmsg.view(batch_size, 1, -1) for fmsg in fmsgs], 1)
    bmsgs = torch.cat([bmsg.view(batch_size, 1, -1) for bmsg in bmsgs], 1)

    bb = bmsgs[:, 2:, :]
    ff = fmsgs[:, :-2, :]
    b0 = logsumexp(bb + ff, 2).view(batch_size, dim_node-1)
    b1 = logsumexp(bb[:, :, :-1] + ff[:, :, 1:], 2).view(
        batch_size, dim_node-1) + node_potentials[:, :-1]

    marginals = create_var(0, batch_size, dim_node)
    marginals[:, :-1] = torch.sigmoid(b1 - b0)

    # Could probably structure things so the Dth var doesn't need to be
    # special-cased.  but this will do for now.  rather than computing
    # a belief at a pairwise potential, we do it at the variable.
    b0_D = fmsgs[:, dim_node-1, 0] + bmsgs[:, dim_node, 0]
    b1_D = fmsgs[:, dim_node-1, 1] + bmsgs[:, dim_node, 1]
    marginals[:, dim_node-1] = torch.sigmoid(b1_D - b0_D)

    return marginals
