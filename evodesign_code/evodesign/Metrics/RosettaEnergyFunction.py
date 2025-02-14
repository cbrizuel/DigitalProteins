from .Metric import Metric
from typing import Optional, List
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd
import pyrosetta





class RosettaEnergyFunction(Metric):

  def __init__(self, column: Optional[str] = None) -> None:
    super().__init__(column)
    self._score_fn = None
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    pdb_path = f'{context.workspace.pdbs_dir}/prot_{data["sequence_id"]}.pdb'
    score = self._ref2015(pdb_path)
    data[self.column_name()] = score
    return data
  


  def _ref2015(self, pdb_path: str) -> float:
    if self._score_fn is None:
      pyrosetta.init()
      self._score_fn = pyrosetta.get_score_function(True)
    pose = pyrosetta.pose_from_pdb(pdb_path)
    score = self._score_fn(pose)
    return score
  
