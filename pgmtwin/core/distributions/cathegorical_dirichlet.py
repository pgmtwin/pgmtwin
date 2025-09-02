import collections

import numpy as np
from scipy import stats


class DirichletMultinomial:
    """
    Class representing a Dirichlet-Multinomial distribution.
    Parameter alpha is a vector of shape (k,) where k is the number of classes.
    In the case of our application, k < N is the number of jumps that the transition
    can do a priori where alpha[0] represents no-jump, alpha[1] 1-step-jump,
    # alpha[2] 2-step jumps,..., N number of state in MDP.
    """

    # __slots__ = ["alpha", "k"]

    def __init__(self, alpha=None):
        if isinstance(alpha, int):
            self.k = alpha
            self.alpha = np.ones(alpha)
        elif len(alpha) > 1:
            self.k = len(alpha)
            self.alpha = np.array(alpha)
        else:
            raise SyntaxError("Argument should be a vector or an int")

    def __repr__(self):
        return f"DirichletMultinomial({self.alpha})"

    def update(self, data: np.ndarray):
        """
        Update the DirichletMultinomial distribution with new counts.
        In particular, it take as input data that is a np.ndarray of shape
        (n_states, n_states) and update the distribution with the counts
        """
        # WARNING: BUG introduction. Nel caso in cui la matrice di transizione
        # sia definita come per i nostri esempi, in cui dallo stato 0 si può saltare
        # in una classe di primo danno qualsiasi, qua risulterà che sono possibili
        # dei salti multipli importanti (anche se non vero).
        # Osservazione: inoltre, la somma degli offset non è al 100% rigorosa.
        # Bisognerebbe inserire un input parameter che ci dica se la matrice è
        # relativa a un azione "danneggiamento" o "riparazione", di conseguenza
        # fare controllo sulla diagonalità superiore/inferiore e mandare warnings.
        counts = {
            i: data.diagonal(offset=i).sum() + data.diagonal(offset=-i).sum()
            for i in range(data.shape[0])
        }
        if isinstance(counts, list):
            counts = collections.Counter(counts)
        if not isinstance(counts, dict):
            raise SyntaxError("Argument should be a dict or a list")
        counts_vec = [counts.get(i, 0) for i in range(self.k)]
        self.alpha = np.add(self.alpha, counts_vec)

    def pdf(self, x):
        diri = stats.dirichlet(self.alpha)
        return diri.pdf(x)

    def cdf(self, weights, x):
        Omega = lambda row: np.dot(weights, row)
        # Sample from Dirichlet posterior
        samples = np.random.dirichlet(self.alpha, 100000)
        # apply sum to sample draws
        W_samples = np.apply_along_axis(Omega, 1, samples)
        # Compute P(W > x)
        return (W_samples > x).mean()

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles) at q.
        Aka Value at Risk.
        :param q: lower tail probability
        :return: quantile corresponding to the lower tail probability q.
        """
        pass
        # return stats.dirichlet.ppf(q, self.alpha)

    def mean(self, n=1):
        return self.alpha * n / (self.alpha.sum())

    @property
    def mode(self):
        return (self.alpha - 1) / (self.alpha.sum() - self.k)

    def posterior(self, weights, lb, ub):
        if lb > ub:
            return 0.0
        return self.cdf(weights, lb) - self.cdf(weights, ub)

    def sample(self, n=1):
        return np.random.dirichlet(self.alpha, n)
