"""
General utilities for the pgmtwin library
"""

from contextlib import contextmanager
import heapq
import inspect
import logging
import re
from typing import List, Tuple, Union

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
import numpy as np

from pgmtwin.core.domain import DiscreteDomain

CMAP_BLUES = ListedColormap(mpl.colormaps["Blues"](np.linspace(0.1, 0.7, 128)))


def get_number_from_text(txt: str, re_group: int = -1) -> int:
    """
    Extracts a numeric value from a string.
    e.g. get_number_from_text("code010_202.ext", re_group=-1) -> 202
    e.g. get_number_from_text("code010_202.ext", re_group=0) -> 10

    Args:
        txt (str): the string to extract the number from
        re_group (int, optional): index of the group to be extracted. Defaults to -1.

    Returns:
        int: the integer value converted from a consecutive group of digits,
    at the requested position in the list of all groups of digits in fname.
    Does not handle minus sign nor decimal point

    Raises:
        ValueError: if no digits are found in the string
    """
    matched = re.findall(r"([0-9]+)", txt)

    if matched:
        try:
            group = matched[re_group]
            return int(group)
        except:
            pass

    return -1


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Get a Boltzmann/Gibbs distribution computed form the given array
    The array is flattenend before the computation

    Args:
        x (np.ndarray): the input array to compute the softmax from

    Returns:
        np.ndarray: the softmax distribution computed from the input array
    """
    x = x.flatten()
    x -= np.max(x)
    return np.exp(x) / sum(np.exp(x))


def normalize_cpd(cpd: np.ndarray, smoothing: float = 0) -> np.ndarray:
    """
    Normalizes a conditional probability distribution to ensure that its columns sum to 1.

    Args:
        cpd (np.ndarray): a 2d array to normalize
        smoothing (float, optional): a quantity to distribute across the elements of each column, to smooth the zero elements

    Returns:
        np.ndarray: the normalized conditional probability distribution
    """
    d = np.sum(cpd + smoothing, axis=0, keepdims=True)
    if not np.allclose(d, 1) or np.any(np.isnan(d) | np.isclose(d, 0)):
        cpd += smoothing + 1e-6
        d = np.sum(cpd, axis=0, keepdims=True)
        cpd /= d
    return cpd


def get_combined_cpd_pair(
    A: np.ndarray, B: np.ndarray, smoothing: float = 0
) -> np.ndarray:
    """
    A and B are two cpd factors resulting in the same variable
    This method assembles the tensor from the 2d observation to the output,
    so that the input variables of the factors are concatenated

    Args:
        A (np.ndarray): the first cpd factor
        B (np.ndarray): the second cpd factor
        smoothing (float, optional): a quantity to distribute across the elements of each column, to smooth the zero elements

    Returns:
        np.ndarray: the combined cpd factor
    """
    C = np.einsum("ij,ik->ijk", A, B).reshape(len(A), -1)
    return normalize_cpd(C, smoothing=smoothing)


def get_combined_cpds(cpds: List[np.ndarray]) -> np.ndarray:
    """
    Takes a list of factor cpds and assembles the tensor
    from the product space of the factors, to the output

    Args:
        cpds (List[np.ndarray]): a list of 2d arrays, each representing a cpd factor

    Returns:
        np.ndarray: the combined cpd factor
    """
    ret = cpds[0]
    for cpd in cpds[1:]:
        ret = get_combined_cpd_pair(ret, cpd)

    return ret


def get_upper_transitions_block(n_states: int, jump_probs: np.ndarray) -> np.ndarray:
    """
    Assembles the transition matrix for a stochastic state regression
    jump_probs[i] is the probability of a transition from state j to j - i

    Args:
        n_states (int): number of states in the system
        jump_probs (np.ndarray): an array of probabilities for each jump

    Returns:
        np.ndarray: a 2d (n_states, n_states) matrix whose columns sum to 1
    """
    ret = np.eye(n_states)

    for j in range(n_states):
        n_elems_actual = min(1 + j, len(jump_probs))

        if n_elems_actual < len(jump_probs):
            jump_probs_actual = jump_probs[: n_elems_actual - 1] + [
                np.sum(jump_probs[n_elems_actual - 1 :])
            ]
        else:
            jump_probs_actual = jump_probs

        ret[j - n_elems_actual + 1 : j + 1, j] = jump_probs_actual[::-1]

    return ret


def get_lower_transitions_block(n_states: int, jump_probs: np.ndarray) -> np.ndarray:
    """
    Assembles the transition matrix for a stochastic state regression
    jump_probs[i] is the probability of a transition from state j to j + i

    Args:
        n_states (int): number of states in the system
        jump_probs (np.ndarray): an array of probabilities for each jump

    Returns:
        np.ndarray: a 2d (n_states, n_states) matrix whose columns sum to 1
    """
    ret = get_upper_transitions_block(n_states, jump_probs)
    ret = np.flipud(np.fliplr(ret))

    return ret


def get_epsilon_greedy_policy(
    action_state_values: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    """
    Generates an epsilon-greedy policy based on the action-state values.
    The highest value action has a probability of 1 - epsilon, while all other actions have a probability of epsilon / n_actions.

    Args:
        action_state_values (np.ndarray): an array of shape (n_actions, n_states) with action-state values
        epsilon (float, optional): the probability threshold for random selection. Defaults to 1e-3.

    Returns:
        np.ndarray: an array of shape (n_actions, n_states) with the epsilon-greedy policy distribution
    """
    n_actions, _ = action_state_values.shape
    ret = np.full_like(action_state_values, fill_value=epsilon / n_actions)

    best_action_idxs = np.argmax(action_state_values, axis=0)

    for s, a in enumerate(best_action_idxs):
        ret[a, s] += 1 - epsilon

    return ret


def get_transition_matrix_from_history(
    n_states: int,
    state_idx_from: np.ndarray,
    state_idx_to: np.ndarray,
) -> np.ndarray:
    """
    Assembles the transition matrix from a history of state transitions.

    Args:
        n_states (int): number of states in the system
        state_idx_from (np.ndarray): history of starting state indices
        state_idx_to (np.ndarray): history of ending state indices

    Returns:
        np.ndarray: a 2d (n_states, n_states) matrix whose columns sum to 1
    """
    ret = np.zeros((n_states, n_states))

    for j, i in zip(state_idx_from, state_idx_to):
        ret[i, j] += 1

    return ret / np.sum(ret)


def get_combined_transitions_from_block(
    transition: np.ndarray, n_dims: int
) -> np.ndarray:
    """
    Assembles the transition matrix for the product of n_dims factor variables,
    each independently transitioning according to transition

    Args:
        transition (np.ndarray): a 2d (n_states, n_states) matrix whose columns sum to 1
        n_dims (int): number of dimensions to combine

    Returns:
        np.ndarray: a 2d (n_states**n_dims, n_states**n_dims) matrix whose columns sum to 1
    """
    n_states = transition.shape[0]

    acc = transition
    for _ in range(n_dims - 1):
        acc = np.tensordot(acc, transition, axes=0)

    # bring the indexing from (i1, i0, j1, j0, ...)
    # to be (i1, j1, ..., i0, j0, ...)
    axes = np.arange(2 * n_dims).reshape(n_dims, 2).T.flatten()
    acc = acc.transpose(*axes)

    return acc.reshape(n_states**n_dims, -1)


def plot_discrete_domain_confusion_matrix(
    state_domain: DiscreteDomain,
    confusion_matrix: np.ndarray,
    ticklabels: List[str] = None,
    normalize: bool = True,
    colorbar: bool = True,
    cmap: Union[str, Colormap] = CMAP_BLUES,
):
    """
    Plots a confusion matrix using the given DiscreteDomain for the tick labels

    Args:
        state_domain (DiscreteDomain): the state domain
        confusion_matrix (np.ndarray): the collected observation counts [true, predicted]
        ticklabels (List[str], optional): the labels for the states, or None to use the values from the domain. Defaults to None.
        normalize (bool, optional): whether to normalize rows to obtain the probability of misprediction. Defaults to True.
        colorbar (bool, optional): whether to draw a colorbar. Defaults to True.
        cmap (Union[str, Colormap], optional): the colormap to use. Defaults to CMAP_BLUES.
    """
    if normalize:
        # normalize rows
        confusion_matrix = normalize_cpd(confusion_matrix.T.copy()).T

        im = plt.imshow(confusion_matrix, cmap=cmap, vmin=0, vmax=1)
    else:
        im = plt.imshow(confusion_matrix, cmap=cmap)

    plt.xlabel("estimated")
    plt.ylabel("true")

    ticks = list(range(len(state_domain)))
    if ticklabels is None:
        ticklabels = list(
            str(s_values) for s_values in state_domain.index2values(ticks)
        )

    plt.xticks(ticks, labels=ticklabels, rotation=90)
    plt.yticks(ticks, labels=ticklabels)

    plt.colorbar(im)
    plt.tight_layout()


def filter_kwargs(func, kwargs: dict):
    """
    Filters a kwargs dictionary to retain only the arguments accepted by the given callable

    Args:
        func (_type_): a callable
        kwargs (dict): a kwargs dictionary

    Returns:
        dict: a kwargs dictionary compatible with func
    """
    sig = inspect.signature(func)
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def sample_joint_distributions(
    n_samples: int,
    pmfs: List[np.ndarray],
    rng: np.random.Generator = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generate at most n_samples indices from the given probability mass functions, in descending order of joint probability
    Returns both the sequence of indices and the corresponding probabilities

    Args:
        n_samples (int): number of samples to draw
        pmfs (np.ndarray): arrays defining the probability mass functions
        rng (np.random.Generator, optional): pseudo-random generator, or None to create a new one. Defaults to None.

    Returns:
        Tuple[List[List[int]], np.ndarray]: list of sampled index records, and probabilities of the corresponding sample
    """
    if rng is None:
        rng = np.random.default_rng()

    pmfs = [pmf / np.sum(pmf) for pmf in pmfs]
    sorters = [
        np.flip(np.argsort(pmf * (1 + 1e-3 * rng.random(len(pmf))))) for pmf in pmfs
    ]
    idxs = (0,) * len(pmfs)

    def __p(idxs):
        return np.prod([pmf[sorter[s]] for pmf, sorter, s in zip(pmfs, sorters, idxs)])

    prioq = [(-__p(idxs), idxs)]
    explored = set([idxs])

    ret = []
    ret_w = []

    for _ in range(n_samples):
        if not prioq:
            break

        p, idxs = heapq.heappop(prioq)

        if np.isclose(p, 0):
            break

        explored.remove(idxs)
        ret.append([sorter[s] for sorter, s in zip(sorters, idxs)])
        ret_w.append(-p)

        for d, pmf in enumerate(pmfs):
            if idxs[d] < len(pmf) - 1:
                iidxs = list(idxs)
                iidxs[d] += 1
                iidxs = tuple(iidxs)

                if iidxs not in explored:
                    heapq.heappush(prioq, (-__p(iidxs), iidxs))
                    explored.add(iidxs)

    return ret, ret_w


class PrefixLogFilter(logging.Filter):
    """
    Helper class to filter log messages with a given prefix
    """

    def __init__(self, msg_prefix: str):
        super().__init__()

        self.msg_prefix = msg_prefix

    def filter(self, record: logging.LogRecord):
        return not record.getMessage().startswith(self.msg_prefix)


@contextmanager
def pgmpy_suppress_cpd_replacement_warning():
    """
    Context manager to filter log messages from pgmpy, about replacing an existing CPD

    Yields:
        PrefixLogFilter: a filter that suppresses messages about replacing existing CPDs
    """
    msg_prefix = "Replacing existing CPD for"
    filter_ = PrefixLogFilter(msg_prefix)

    logger = logging.getLogger("pgmpy")
    logger.addFilter(filter_)

    try:
        yield filter_
    finally:
        logger.removeFilter(filter_)


@contextmanager
def pgmpy_suppress_cpd_sum_warning():
    """
    Context manager to filter log messages from pgmpy, about cpd normalization

    Yields:
        PrefixLogFilter: a filter that suppresses messages about CPD values not summing to 1
    """
    msg_prefix = "Probability values don't exactly sum to 1"
    filter_ = PrefixLogFilter(msg_prefix)

    logger = logging.getLogger("pgmpy")
    logger.addFilter(filter_)

    try:
        yield filter_
    finally:
        logger.removeFilter(filter_)


class SingletonMeta(type):
    """
    Metaclass for implementing Singleton pattern.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
