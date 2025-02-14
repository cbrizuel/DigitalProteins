from .Metric import Metric
from typing import List, Optional
from Bio.PDB.Atom import Atom
import itertools
import numpy as np
import pandas as pd
from ..Context import Context





class ContactMapRms(Metric):
  
  def __init__(self, column: Optional[str] = None) -> None:
    super().__init__(column)
    self._ref_map = None
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    # TODO: guardar los mapas de contacto en un archivo
    data[self.column_name()] = self._contact_map_rms(backbone, context.ref_backbone)
    return data
  


  def _contact_map(self, backbone: List[Atom]):
    ca_atoms = [ 
      atom 
      for atom in backbone 
      if atom.get_name() == "CA" 
    ]
    map_values = np.array([ 
      a - b 
      for (a, b) in itertools.combinations(ca_atoms, 2) 
    ])
    return map_values
  


  def _contact_map_rms(self, 
                       backbone: List[Atom],
                       ref_backbone: List[Atom]
                       ) -> float:
    if self._ref_map is None:
      self._ref_map = self._contact_map(ref_backbone)
    model_map = self._contact_map(backbone)
    # computing the mean over all individual values is the equivalent as computing
    # the weighted mean of the means of each value group
    rms = np.sqrt(np.mean((self._ref_map - model_map)**2))
    return rms
