from abc import ABC, abstractmethod
from ..SettingsRetrievable import SettingsRetrievable
from typing import Optional, List
import pandas as pd
from Bio.PDB.Atom import Atom
from ..Context import Context





class Metric(SettingsRetrievable, ABC):

  def _params(self) -> dict:
    return { 'column': self._column_name }



  def column_name(self) -> str:
    return self._column_name
  


  def __init__(self, column: Optional[str] = None) -> None:
    """
    Some metric or function to be computed for every sequence in a population
    in order to compute their fitness value.

    Parameters
    ----------
    column : str, optional
        The name that should identify the values of this metric in the CSV file
        storing the population data. If `None`, then the class name will be 
        used. Default is `None`.
    """
    super().__init__()
    self._column_name = column
    if column is None or len(column) == 0:
      self._column_name = self._class_name()

  

  def __call__(self, 
               backbone: List[Atom],
               data: pd.Series,
               context: Context
               ) -> pd.Series:
    """
    Computes the current metric for a given model backbone or sequence.

    Parameters
    ----------
    backbone : List[Bio.PDB.Atom.Atom]
        The backbone for which the metric will be computed.
    data : pandas.Series
        Other data associated with the given backbone. Typically consists of the
        amino acid sequence and other metrics computed from said sequence. 
    context : Context
        The context data used by the calling evolutionary algorithm.

    Returns
    -------
    pandas.Series
        The original data series with added columns to include the computed metric 
        value.
    """
    if self.column_name() in data.index:
      return data
    return self._compute_values(backbone, data, context)
  


  @abstractmethod
  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    raise NotImplementedError
