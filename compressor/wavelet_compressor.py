import numpy as np
import pywt

from compressor.compressor import Compressor

class WaveletCompressor(Compressor):
    def __init__(self, threshold, wavelet='haar'):
        super().__init__(threshold)
        self._wavelet = wavelet

    def compress(self, data: np.ma.masked_array) -> np.ma.masked_array:
        # Perform wavelet transform on the data
        coeff_arr, slices = pywt.wavedecn(data, self._wavelet, level=3)
        thresholded_coeff_arr = pywt.threshold(coeff_arr, 0.35, mode='hard')
        coeffs = pywt.array_to_coeffs(thresholded_coeff_arr, slices, 'wavedecn')
        
        return coeffs

    def decompress(self, compressed_data: np.ma.masked_array) -> np.ma.masked_array:
        # Reconstruct the wavelet coefficients
        coeffs = compressed_data

        # Perform inverse wavelet transform
        decompressed_data = pywt.waverecn(coeffs, wavelet=self._wavelet, level=3)

        return decompressed_data