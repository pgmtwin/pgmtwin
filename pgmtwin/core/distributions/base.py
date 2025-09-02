"""Base class for all conjugate models and distributions for Bayesian inference."""

from abc import ABC


class BayesModel(ABC):
    def update(self, *args, **kwargs):
        """Update the posterior parameters given the data."""
        pass

    def pdf(self, *args, **kwargs):
        """Probability density function."""
        pass

    def cdf(self, *args, **kwargs):
        """Cumulative distribution function."""
        pass

    def ppf(self, *args, **kwargs):
        """Percent point function (inverse of cdf — percentiles)."""
        pass

    def posterior(self, *args, **kwargs):
        """Posterior distribution."""
        pass

    def mean(self, *args, **kwargs):
        """Mean of the posterior distribution."""
        pass

    @property
    def mode(self):
        """Maximum a posteriori (MAP)."""
        pass

    def sample(self, output_parameter):
        """Sample from the posterior distribution."""
        pass
