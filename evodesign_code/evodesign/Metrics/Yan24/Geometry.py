from ..Metric import Metric
from ..Normalization.Reciprocal import Reciprocal
from ..Rmsd import Rmsd
from ..Gdt import Gdt
from ..ContactMapRms import ContactMapRms
from ..TMScore import TMScore
from ...Context import Context
from typing import Optional, List, Dict
from Bio.PDB.Atom import Atom
import pandas as pd





class Geometry(Metric):
  
  def _params(self) -> dict:
    params = super()._params()
    params['rmsd_metric'] = self._rmsd_metric.settings()
    params['gdt_metric'] = self._gdt_metric.settings()
    params['tmscore_metric'] = self._tmscore_metric.settings()
    params['cm_rms_metric'] = self._cm_rms_metric.settings()
    return params
  


  def __init__(self, 
               rmsd_metric: Optional[Rmsd] = None,
               gdt_metric: Optional[Gdt] = None,
               tmscore_metric: Optional[TMScore] = None,
               cm_rms_metric: Optional[ContactMapRms] = None, 
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    if rmsd_metric is None:
      rmsd_metric = Rmsd()
    self._rmsd_metric = rmsd_metric
    if gdt_metric is None:
      gdt_metric = Gdt(rmsd_metric=self._rmsd_metric)
    self._gdt_metric = gdt_metric
    if tmscore_metric is None:
      tmscore_metric = TMScore(rmsd_metric=self._rmsd_metric)
    self._tmscore_metric = tmscore_metric
    if cm_rms_metric is None:
      self._cm_rms_metric = Reciprocal(ContactMapRms())
    else:
      self._cm_rms_metric = Reciprocal(cm_rms_metric)
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> Dict[str, float]:
    data = self._rmsd_metric(backbone, data, context)
    data = self._gdt_metric(backbone, data, context)
    data = self._cm_rms_metric(backbone, data, context)
    data = self._tmscore_metric(backbone, data, context)
    a = self._rmsd_metric.column_name()
    b = self._gdt_metric.column_name()
    c = self._cm_rms_metric.column_name()
    d = self._tmscore_metric.column_name()
    data[self.column_name()] = data[a] + data[b] + data[c] + data[d]
    return data
  