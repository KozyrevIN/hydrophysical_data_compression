from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from numpy.ma import masked_array

class Compressor(ABC):
    def __init__(self, threshold: float):
        """
        Initialize the Compressor with a threshold and dimensions.

        :param threshold: The threshold value for compression.
        :param dimensions: An object containing x, y, z dimensions.
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

