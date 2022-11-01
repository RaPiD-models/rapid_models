from typing import Union

import gpytorch
import gpytorch.constraints
import torch


# @TODO: This function is nowhere called.
#        If we keep it, we should add type hints and possibly a unit test.
#        @AGRE / @ELD: Delete?
#        CLAROS, 2022-11-01
def optim_step(model, loss_function, optimizer):
    """
    Return current loss and perform one optimization step
    """
    # zero the gradients
    optimizer.zero_grad()

    # compute output from model
    output = model(model.train_inputs[0])

    # calc loss and backprop gradients
    loss = -loss_function(output, model.train_targets)
    loss.backward()
    optimizer.step()

    return loss.item()


def gpytorch_kernel_Matern(
    var: torch.Tensor,
    ls: torch.Tensor,
    nu: float = 2.5,
    lengthscale_constraint: Union[gpytorch.constraints.Interval, None] = None
) -> gpytorch.kernels.Kernel:
    """
    Return a Matern kernel with specified kernel variance (var) and lengthscales (ls)
    """
    lengthscale_constraint = lengthscale_constraint or gpytorch.constraints.Positive(
    )
    ker_mat = gpytorch.kernels.MaternKernel(
        nu=nu,
        ard_num_dims=len(ls),
        lengthscale_constraint=lengthscale_constraint)
    ker_mat.lengthscale = ls
    ker = gpytorch.kernels.ScaleKernel(ker_mat)
    ker.outputscale = var

    return ker


def gpytorch_mean_constant(val: float,
                           fixed: bool = True) -> gpytorch.means.ConstantMean:
    """
    Return a constant mean function

    fixed = True -> Do not update mean function during training
    """
    mean = gpytorch.means.ConstantMean()
    mean.initialize(constant=val)
    assert isinstance(mean.constant, torch.Tensor)
    mean.constant.requires_grad = not fixed

    return mean


def gpytorch_likelihood_gaussian(
        variance: torch.Tensor,
        variance_lb: Union[float, torch.Tensor] = 1e-6,
        fixed: bool = True) -> gpytorch.likelihoods.Likelihood:
    """
    Return a Gaussian likelihood

    fixed = True -> Do not update during training
    variance_lb = lower bound
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(variance_lb))
    likelihood.initialize(noise=variance)
    # @TODO: Base type of likelihood is gpytorch.Module, not torch.Tensor
    #        Natively, hence, likelihood does not have an attribute 'requires_grad'.
    #        What the following line effectively does is to dynamically
    #        add an attribute with name='requires_grad' to likelihood and assign it a boolean value.
    #        @AGRE / @ELD: Is this really what you intended, and is it necessary?
    #        CLAROS, 2022-11-01
    likelihood.requires_grad = not fixed

    return likelihood
