"""
Abstract and base implementations of components for the addition of noise to the observations of a PhysicalAsset
"""

import numpy as np


class BaseObservationNoiseComponent:
    """
    Abstract class for a noise component
    """

    def apply_noise(
        self,
        state: np.ndarray,
        observation: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Returns the transformed observation with added noise

        Args:
            state (np.ndarray): the current state of the asset
            observation (np.ndarray): the current observation value record
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: the observation with added noise
        """
        return observation


class GaussianNoiseComponent(BaseObservationNoiseComponent):
    """
    Class to add Gaussian distributed noise with given mean and stddev
    """

    def __init__(self, mean: float, std: float):
        """
        Sets the mean and stddev of the Gaussian noise that will be added to observations

        Args:
            mean (float): the mean of the Gaussian noise
            std (float): the standard deviation of the Gaussian noise
        """
        self.mean = mean
        self.std = std

    def apply_noise(
        self,
        state: np.ndarray,
        observation: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        return observation + rng.normal(self.mean, self.std, observation.shape)


class SNRGaussianNoiseComponent(BaseObservationNoiseComponent):
    """
    Class to add noise zero-mean Gaussian noise, with the stddev depending on the observations values
    Has an effect only when applied to a multi-record observation
    """

    def __init__(self, signal2noise_ratio: float):
        """
        Sets the signal-to-noise ratio that will be used to generate the noise

        Args:
            signal2noise_ratio (float): the signal-to-noise ratio to use for the noise generation
        """
        self.signal2noise_ratio = signal2noise_ratio

    def apply_noise(
        self,
        state: np.ndarray,
        observation: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if observation.ndim == 1:
            return observation

        if rng is None:
            rng = np.random.default_rng()

        signal_var = np.var(observation, axis=0, keepdims=True)
        noise_std = np.sqrt(signal_var / self.signal2noise_ratio)

        return observation + rng.normal(
            loc=0.0, scale=noise_std, size=observation.shape
        )
