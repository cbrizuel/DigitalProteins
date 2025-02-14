from abc import ABC, abstractmethod
from typing import List, Optional
from ..SettingsRetrievable import SettingsRetrievable
import pandas as pd
import numpy.typing as npt
import numpy as np





class FitnessFunction(SettingsRetrievable, ABC):

  def column_name(self) -> str:
    return self._column_name
  


  def _params(self) -> dict:
    return {
      'column': self._column_name,
      'upper_bound': self.upper_bound,
      'metric_columns': self._metric_columns
    }
  


  def __init__(self, 
               upper_bound: float,
               metric_columns: List[str],
               column: Optional[str] = None
               ) -> None:
    super().__init__()
    self.upper_bound = upper_bound
    self._metric_columns = metric_columns
    self._column_name = column
    if column is None or len(column) == 0:
      self._column_name = self._class_name()
  


  def __call__(self, data: pd.Series) -> pd.Series:
    """
    Computes the fitness value.

    Parameters
    ----------
    data : pandas.Series
        The population data corresponding to the individual which fitness 
        value is being computed. It is assumed that this data also includes the values 
        of the metrics required by the current fitness function.

    Returns
    -------
    pandas.Series
        The original data series with added columns to include the computed fitness 
        value.
    """
    metric_values = data[self._metric_columns].to_numpy()
    result = self.compute_fitness(metric_values)
    data[self.column_name()] = result
    return data
  


  @abstractmethod
  def compute_fitness(self, metrics: npt.NDArray[np.float64]) -> float:
    raise NotImplementedError
