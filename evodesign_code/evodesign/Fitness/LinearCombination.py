from .FitnessFunction import FitnessFunction
from typing import List, Optional
import numpy as np
import numpy.typing as npt





class LinearCombination(FitnessFunction):
  
  def _params(self) -> dict:
    params = super()._params()
    params['coefficients'] = self._coefficients.tolist()
    return params
  


  def __init__(self, 
               upper_bound: float,
               metric_columns: List[str],
               coefficients: Optional[List[float]] = None,
               column: Optional[str] = None
               ) -> None:
    """
    Computes the fitness of a given individual as a linear combination of the
    specified terms.

    Parameters
    ----------
    upper_bound : float
        The maximum allowed fitness value before triggering a stop condition
        in the evolutionary algorithm.
    metric_columns : List[str]
        The names that identify the metrics required for computing the fitness value in
        the CSV storing the population data.
    coefficients : List[float], optional
        The coefficients for each individual metric. If no coefficients are
        specified, all metrics are assigned a coefficient of 1.0, turning this
        fitness function into a simple addition of the metrics. Default is `None`.
    column : str, optional
        The name that should identify the values of this metric in the CSV file
        storing the population data. If `None`, then the class name will be 
        used. Default is `None`.

    Raises
    ------
    RuntimeError
        If the length of `metric_columns` and `coefficients` are not the same.
    """
    super().__init__(upper_bound, metric_columns, column)
    if coefficients is None or len(coefficients) == 0:
      coefficients = len(self._metric_columns) * [ 1.0 ]
    if len(coefficients) != len(self._metric_columns):
      raise RuntimeError
    self._coefficients = np.array(coefficients)
  


  def compute_fitness(self, metrics: npt.NDArray[np.float64]) -> float:
    """
    Computes a linear combination of the given metric values using the given 
    coefficients.

    Parameters
    ----------
    metrics : numpy.typing.NDArray[numpy.float64]
        The metric values to be used to compute the fitness value.

    Returns
    -------
    float
        The computed fitness value.
    """
    return (self._coefficients * metrics).sum()
