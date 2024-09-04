import numpy as np
import pywt

from compressor.compressor import Compressor

class WaveletCompressor(Compressor):
    def __init__(self, threshold, wavelet = 'haar', interpolation = 'none'):
        super().__init__(threshold)
        self._wavelet = wavelet
        self._interpolation = interpolation

    def compress(self, data: np.ma.masked_array) -> np.ma.masked_array:
        # Perform wavelet transform on the data
        coeff_arr, slices = pywt.coeffs_to_array(pywt.wavedecn(data, self._wavelet, level=3))
        thresholded_coeff_arr = pywt.threshold(coeff_arr, 0.34, mode = 'hard')
        coeffs = pywt.array_to_coeffs(thresholded_coeff_arr, slices, 'wavedecn')
        
        return coeffs

    def decompress(self, compressed_data: np.ma.masked_array) -> np.ma.masked_array:
        # Reconstruct the wavelet coefficients
        coeffs = compressed_data

        # Perform inverse wavelet transform
        decompressed_data = pywt.waverecn(coeffs, wavelet=self._wavelet)

        return decompressed_data
    
    def compression_coefficient(self, data, compress_data) -> float:
        return (compress_data.shape[0] * compress_data.shape[1]) / np.count_nonzero(compress_data) 