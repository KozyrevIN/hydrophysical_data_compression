from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


def error(data : ndarray, compress_data = ndarray, norm = 'C') -> float:
    if (norm == 'C'):
        return np.linalg.norm(data - compress_data, ord=np.inf)
    elif (norm == 'L2'):
        return np.linalg.norm(data - compress_data, ord=2)
    return None

class Compressor(ABC):
    def __init__(self, threshold: float):
        """
        Initialize the Compressor with a threshold and dimensions.

        :param threshold: The threshold value for compression.
        """
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Getter for the threshold value."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        """Setter for the threshold value."""
        if value < 0:
            raise ValueError("Threshold must be a non-negative value.")
        self._threshold = value

    @abstractmethod
    def compression_coefficient(self, data, compress_data) -> float:
        """
        Abstract method for compressing data.

        :param data: The initial data to be compressed.
        :param compress_data: The compressed.
        :return: The coefficient of compression.
        """
        pass

    @abstractmethod
    def compress(self, data: ndarray, mask: ndarray = []):
        """
        Abstract method for compressing data.

        :param data: The data to be compressed.
        :param mask: Mask indicating land area.
        :return: The compressed data.
        """
        pass

    @abstractmethod
    def decompress(self, compressed_data, mask: ndarray = []):
        """
        Abstract method for decompressing data.

        :param compressed_data: The compressed data to be decompressed.
        :param mask: Mask indicating land area.
        :return: The decompressed data.
        """
        pass

    @abstractmethod
    def compress_coefficient(self, data: ndarray, mask: ndarray = []):
        """
        Abstract method for compressing data.

        :param data: The data to be compressed.
        :param mask: Mask indicating land area.
        :return: The compressed data.
        """
        pass