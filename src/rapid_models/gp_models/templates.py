import contextlib
from typing import Any, Tuple, Union

import gpytorch
import torch
from nptyping import NDArray


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Model for standard GP regression
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
        path: str = '',
        name: str = '',
    ):
        # Note: Overwriting the declaration of self.likelihood is necessary to make explicit to code linters
        #       that likelihood is not optional in our implementation, i.e. likelihood cannot be None.
        #       (This is different from the ExactGP base class implementation where likelihood can also be None.)
        self.likelihood: gpytorch.likelihoods.Likelihood

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        assert self.likelihood is not None

        # For saving and loading
        self.path = path
        self.name = name
        self.param_fname = self.path + self.name + ".pth"

        # Mean and covariance functions
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self,
                x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval_mode(self):
        """
        Set model in evaluation mode
        """
        self.eval()
        self.likelihood.eval()

    def train_mode(self):
        """
        Set in training mode
        """
        self.train()
        self.likelihood.train()

    def predict(
        self,
        x: torch.Tensor,
        latent: bool = True,
        CG_tol: float = 0.1,
        full_cov: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean and covariance at x

        Input:
        x         -      tensor of size dim * N containing N inputs
        latent    -      latent = True ->  using latent GP
                         latent = False -> using observed GP (incl. likelihood)
        CG_tol    -      Conjugate Gradient tolerance for evaluation
        full_cov  -      full_cov = False -> Return only diagonal (variances)

        Output:
        mean and covariance
        """

        mean: torch.Tensor
        var: torch.Tensor
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(CG_tol):

            # Latent distribution
            dist: torch.distributions.Distribution = self.__call__(x)

            # Observational distribution
            if not latent:
                _dist = self.likelihood(dist)
                if isinstance(_dist, torch.distributions.Distribution):
                    dist = _dist

            # Extract mean and covariance
            assert isinstance(dist, gpytorch.distributions.MultivariateNormal)
            # if isinstance(dist, gpytorch.distributions.MultivariateNormal):
            mean = dist.mean.cpu()
            var = dist.covariance_matrix.cpu(  # type: ignore
            ) if full_cov else dist.variance.cpu()

        return mean, var

    def print_parameters(self):
        """
        Print actual (not raw) parameters
        """
        _constant_mean: Union[torch.Tensor, str] = '--'
        with contextlib.suppress(Exception):
            _constant_mean = self.mean_module.constant.item()  # type: ignore

        _noise: Union[torch.Tensor, str] = '--'
        with contextlib.suppress(Exception):
            _noise = self.likelihood.noise_covar.noise.item()  # type: ignore

        _lengthscale: Union[NDArray[Any, Any], str] = '--'
        with contextlib.suppress(Exception):
            _lengthscale = self.covar_module.base_kernel.lengthscale.detach(  # type: ignore
            ).numpy()[0]

        _outputscale: Union[torch.Tensor, str] = '--'
        with contextlib.suppress(Exception):
            _outputscale = self.covar_module.outputscale.item()  # type: ignore

        print("{:50} {}".format("Constant mean", _constant_mean))
        print("{:50} {}".format("Likelihood noise variance", _noise))
        print("{:50} {}".format("Kernel lengthscale", _lengthscale))
        print("{:50} {}".format("Kernel outputscale (variance)", _outputscale))

    def save(self):
        """
        Save GP model parameters to self.path
        """
        print("Saving model to: ", self.param_fname)
        torch.save(self.state_dict(), self.param_fname)

    def load(self):
        """
        Load GP model parameters from self.path
        """
        print("Loading model from: ", self.param_fname)
        self.load_state_dict(torch.load(self.param_fname))
