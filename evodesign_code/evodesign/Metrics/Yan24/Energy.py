from ..Metric import Metric
from ..RosettaEnergyFunction import RosettaEnergyFunction
from .Geometry import Geometry
from ...Context import Context
from typing import Optional, List, Dict
import numpy as np
from Bio.PDB.Atom import Atom
import pandas as pd






class Energy(Metric):
  
  def _params(self) -> dict:
    params = super()._params()
    params['weight'] = self._weight
    params['geometry_metric'] = self._geometry_metric.settings()
    params['energy_metric'] = self._energy_metric
    return params
  


  def __init__(self, 
               weight: float,
               geometry_metric: Optional[Geometry] = None,
               energy_metric: Optional[RosettaEnergyFunction] = None,
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    self._weight = weight
    if geometry_metric is None:
      geometry_metric = Geometry()
    self._geometry_metric = geometry_metric
    if energy_metric is None:
      energy_metric = RosettaEnergyFunction()
    self._energy_metric = energy_metric
    self._ref_energy = None
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> Dict[str, float]:
    if self._ref_energy is None:
      import pyrosetta
      pose = pyrosetta.pose_from_pdb(context.workspace.target_pdb_path)
      self._ref_energy = self._energy_metric.score_fn(pose)
    data = self._geometry_metric(backbone, data, context)
    data = self._energy_metric(backbone, data, context)
    f1 = data[self._geometry_metric.column_name()]
    e = data[self._energy_metric.column_name()]
    total = f1 / 4 + 1 / np.exp(abs(self._ref_energy - e) / self._weight)
    data[self.column_name()] = total
    return data
  