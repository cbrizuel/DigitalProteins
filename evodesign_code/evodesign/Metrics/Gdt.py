from .Metric import Metric
from typing import List, Optional, Dict
from .Rmsd import Rmsd
import numpy as np
from Bio.PDB.Atom import Atom
import pandas as pd
from ..Context import Context





class Gdt(Metric):
  
  def _params(self) -> dict:
    params = super()._params()
    params['cutoffs'] = self._cutoffs
    params['rmsd_metric'] = self._rmsd_metric.settings()
    return params
  


  def __init__(self,
               cutoffs: List[float] = [ 1.0, 2.0, 4.0, 8.0 ],
               rmsd_metric: Optional[Rmsd] = None,
               column: Optional[str] = None
               ) -> None:
    """
    Computes the GDT value between the model backbone and a reference backbone
    after superimposing the former into the latter. Both backbones must contain 
    equal number of atoms.

    Parameters
    ----------
    cutoffs : List[float], optional
        The cut-off distances used for computing the GDT. Default value is 
        `[ 1.0, 2.0, 4.0, 8.0 ]`
    """
    super().__init__(column)
    self._cutoffs = cutoffs
    if rmsd_metric is None:
      rmsd_metric = Rmsd()
    self._rmsd_metric = rmsd_metric
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    # superimpose the structures if not already superimposed
    data = self._rmsd_metric(backbone, data, context)
    data[self.column_name()] = self._gdt(backbone, context.ref_backbone)
    return data



  def _gdt(self, 
           backbone: List[Atom], 
           ref_backbone: List[Atom]
           ) -> float:
    # assuming the structures are already superimposed
    distances = np.array([ 
      a - b 
      for a, b in zip(backbone, ref_backbone) 
    ])
    gdt = np.mean([
      np.mean([ d <= c for d in distances ])
      for c in self._cutoffs
    ])
    return gdt
  