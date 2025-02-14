from .FitnessFunction import FitnessFunction
from ..Metrics.Metric import Metric
import numpy as np
from typing import List, Optional
import pandas as pd
import numpy.typing as npt





class WeightedMean(FitnessFunction):
  
  def _params(self) -> dict:
    params = super()._params()
    params['weights'] = self._weights
    return params
  


  def __init__(self, 
               upper_bound: float,
               metric_columns: List[Metric],
               weights: Optional[List[float]] = None,
               column: Optional[str] = None
               ) -> None:
    """
    Computes the fitness of a given individual as the weighted mean of the 
    specified terms.

    Parameters
    ----------
    upper_bound : float
        The maximum allowed fitness value before triggering a stop condition
        in the evolutionary algorithm.
    metric_columns : List[str]
        The names that identify the metrics required for computing the fitness value in
        the CSV storing the population data.
    weights : Optional[List[float]], optional
        The weights for each individual term. If no weights are specified,
        equal weights are used for all terms, turning this fitness function 
        into a simple mean.
    column : str, optional
        The name that should identify the values of this metric in the CSV file
        storing the population data. If `None`, then the class name will be 
        used. Default is `None`.

    Raises
    ------
    RuntimeError
        If the length for `metric_columns` and `weights` are not the same.
    """
    super().__init__(upper_bound, metric_columns, column)
    if weights is None or len(weights) == 0:
      weights = len(self._metric_columns) * [ 1.0 ]
    if len(weights) != len(self._metric_columns):
      raise RuntimeError
    self._weights = weights
  


  def compute_fitness(self, metrics: npt.NDArray[np.float64]) -> float:
    """
    Computes a weighted mean of the given metric values using the given weights.

    Parameters
    ----------
    metrics : numpy.typing.NDArray[numpy.float64]
        The metric values to be used to compute the fitness value.

    Returns
    -------
    float
        The computed fitness value.
    """
    return np.average(metrics, weights=self._weights)
