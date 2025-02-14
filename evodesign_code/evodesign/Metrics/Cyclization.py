from .Metric import Metric
from typing import Optional, List, Dict
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd





class Cyclization(Metric):

  def __init__(self, column: Optional[str] = None) -> None:
    super().__init__(column)



  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    # distance between the N atom of the first residue and the C 
    # atom of the last
    data[self.column_name()] = self._cyclization(backbone)
    return data



  def _cyclization(self, backbone: List[Atom]) -> float:
    return backbone[0] - backbone[-2]
