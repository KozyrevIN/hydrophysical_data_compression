import numpy as np
import pywt
import warnings
import math

def masked_dot(vec_1, vec_2):
    return np.sum(vec_1.data * (~vec_1.mask) * vec_2)

#realization of wavelet interpolation
def transfer_projection_2D(transfer_from, funk, transfer_to):
    funk_length = funk.shape[0]
    size = transfer_from.shape[0]
    num_funks = size // funk_length
    funk_norm_squared = np.linalg.norm(funk, 2)**2

    horyzontal_coeffs = np.ndarray((size, num_funks))
    for i in range(size):
        for j in range(num_funks):
            slice = transfer_from[i, j * funk_length : (j + 1) * funk_length]
            masked = np.count_nonzero(slice.mask)
            if masked < funk_length:
                horyzontal_coeffs[i, j] = masked_dot(slice, funk) * funk_length / (funk_length - masked) / funk_norm_squared
            else:
                horyzontal_coeffs[i, j] = 0
    
    coeffs = np.ndarray((num_funks, num_funks))
    for j in range(num_funks):
        for i in range(num_funks):
            coeffs[i, j] = np.dot(horyzontal_coeffs[i * funk_length : (i + 1) * funk_length, j], funk) / funk_norm_squared
    
    for idx, coeff in np.ndenumerate(coeffs):
        for i in range(funk_length):
            for j in range(funk_length):
                transfer_from[idx[0] * funk_length + i, idx[1] * funk_length + j] -= coeff * funk[i] * funk[j]
                transfer_to[idx[0] * funk_length + i, idx[1] * funk_length + j] += coeff * funk[i] * funk[j]

def wavelet_interpolation(data: np.ma.masked_array, wavelet: str):
    #detecting data shape
    max_level = math.floor(math.log2(data.shape[0]))

    #iteratively transfering approximation components of data into approximated data
    interpolated_data = np.zeros((data.shape[0], data.shape[1]))
    for level in range(max(max_level - 2, 1), 0, -1):
        [phi, _, _] = pywt.Wavelet(wavelet).wavefun(level)
        transfer_projection_2D(data, phi[1:-1], interpolated_data)
    
    interpolated_data += data
    
    return interpolated_data


from compressor.compressor import Compressor

class WaveletCompressor(Compressor):
    def __init__(self, threshold, wavelet = 'haar', interpolation = 'none'):
        super().__init__(threshold)
        self._wavelet = wavelet
        self._interpolation = interpolation

    def compress(self, data: np.ndarray, mask: np.ndarray = []) -> np.ma.masked_array:
        data_to_compress = np.ma.masked_array(data.copy(), mask.copy(), fill_value = 0)

        # Inerpolate the temperature into the land area
        if ~(self._interpolation == 'none'):
            if self._interpolation == 'wavelet':
                interpolated_data = wavelet_interpolation(data_to_compress, self._wavelet)
            elif self._interpolation == 'linear':
                print('aboba')
            else:
                warnings.warn('Specified interpolation if not one of the defined ones. Continuing without interpolation.')
                interpolated_data = data_to_compress
        else:
            interpolated_data = data_to_compress

        # Perform wavelet transform on the data
        coeff_arr, slices = pywt.coeffs_to_array(pywt.wavedecn(interpolated_data.data, self._wavelet, level = 3))

        def approximation_error(coeff_threshold):
            thresholded_coeff_arr = pywt.threshold(coeff_arr, coeff_threshold, mode = 'hard')
            return np.max(self.decompress(pywt.array_to_coeffs(coeff_arr - thresholded_coeff_arr, slices, 'wavedecn')) * ~mask)

        def bisect_max_x(f, threshold, a, b, epsilon = 0.01):
            while b - a > epsilon:
                c = (a + b) / 2
                if f(c) < threshold:
                    a = c
                else:
                    b = c
            return a

        coeff_threshold = bisect_max_x(approximation_error, self._threshold, 0, 1)
        thresholded_coeff_arr = pywt.threshold(coeff_arr, coeff_threshold, mode = 'hard')
        coeffs = pywt.array_to_coeffs(thresholded_coeff_arr, slices, 'wavedecn')
        
        return coeffs

    def decompress(self, compressed_data: np.ma.masked_array) -> np.ma.masked_array:
        # Reconstruct the wavelet coefficients
        coeffs = compressed_data

        # Perform inverse wavelet transform
        decompressed_data = pywt.waverecn(coeffs, wavelet = self._wavelet)

        return decompressed_data
    
    def compression_coefficient(self, data, compress_data) -> float:
        return (compress_data.shape[0] * compress_data.shape[1]) / np.count_nonzero(compress_data) 