from .Rastrigin2 import Rastrigin2
import numpy as np
import numpy.typing as npt





class Quadratic(Rastrigin2):

    def _compute_rastrigin_value(self, x: npt.NDArray[np.float64]) -> float:
        rastrigin = lambda x_i: x_i ** 2
        sigma = np.sum(np.vectorize(rastrigin)(x))
        return -1.0 * sigma # queremos el valor negativo
