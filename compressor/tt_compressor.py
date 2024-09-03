from .compressor import Compressor
import numpy as np
from numpy import ndarray
from numpy.linalg import svd

class TTCompressor(Compressor):
    def __init__(self, treshold, shape = [4]*12):
        super().__init__(treshold)
        self.shape = shape

    def compression_coefficient(self, data : list[np.ndarray], compress_data : list[np.ndarray]) -> float:
        return sum(core.size for core in data) / sum(d.size for d in compress_data)
        

    def compress(self, data: ndarray, mask: ndarray = [], max_rank = 1000000):
        """
        Abstract method for compressing data.

        :param data: The data to be compressed.
        :param mask: Mask indicating land area.
        :return: The compressed data.
        """
        assert(np.prod(data.shape) == np.prod(self.shape))
        tensor_data = data.reshape(self.shape)
        return self.__tt_svd(tensor_data, max_rank)

    def decompress(self, compressed_data, mask: ndarray = [], shape : tuple = (128, 128, 1024)):
        """
        Abstract method for decompressing data.

        :param compressed_data: The compressed data to be decompressed.
        :param mask: Mask indicating land area.
        :return: The decompressed data.
        """
        tensor_data = self.__reconstruct_tt(compressed_data)
        return tensor_data.reshape(shape)
        

    def __tt_svd(self, tensor : ndarray, max_rank : int):
        """
        Performs tensor decomposition into a tensor train with rank truncation.
        
        Parameters:
        - tensor: numpy ndarray, a multidimensional tensor for decomposition.
        - max_rank: int, the maximum allowed rank at each step of the decomposition.
        
        Returns:
        - cores: A list of numpy arrays representing the cores of a tensor train.
        - ranks: A list of ranks used for each core.
        """
        shape = tensor.shape
        num_dims = len(shape)
        ranks = [1]
        cores = []
        
        for i in range(num_dims - 1):
            tensor = tensor.reshape((ranks[-1] * shape[i], -1))
            U, S, Vt = svd(tensor, full_matrices=False)
            rank = min(max_rank, np.sum(S > 1e-6))
            ranks.append(rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            
            core = U.reshape((ranks[-2], shape[i], rank))
            cores.append(core)
            
            tensor = np.dot(np.diag(S), Vt)
        
        cores.append(tensor.reshape((ranks[-1], shape[-1], 1)))
        
        return cores, ranks
    
    def __reconstruct_tt(self, cores):
        """
        Restores the original tensor from its representation as a tensor train.
        
        Parameters:
        - cores: A list of numpy arrays representing the cores of a tensor train.
        
        Returns:
        - tensor: numpy ndarray, the restored tensor.
        """
        tensor = cores[0]
        for i in range(1, len(cores)):
            tensor = np.tensordot(tensor, cores[i], axes=[-1, 0])
        tensor = tensor.squeeze()  
        return tensor