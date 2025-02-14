from .Metric import Metric
from .ESM2Descriptors import ESM2Descriptors
import numpy as np
from typing import Optional, List
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd





class ESM2DescriptorsRms(Metric):

  def _params(self) -> dict:
    params = super()._params()
    params['esm2_metric'] = self._esm2_metric.settings()
    return params
  


  def __init__(self, 
               esm2_metric: Optional[ESM2Descriptors] = None,
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    if esm2_metric is None:
      esm2_metric = ESM2Descriptors()
    self._esm2_metric = esm2_metric 
    self._ref_vectors = None
  


  def _compute_values(self,
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    # TODO cargar los descriptores del archivo almacenado si este ya existe
    if self._ref_vectors is None:
      # compute the ESM2 descriptors
      self._ref_vectors = self._esm2_metric.compute_descriptor_vectors(context.ref_sequence)
      txt_path = f'{context.workspace.esm2_dir}/reference.txt'
      self._esm2_metric.save_descriptor_vectors_txt(self._ref_vectors, txt_path)
    
    # compute the ESM2 descriptors for the model
    data = self._esm2_metric(backbone, data, context)
    c = self._esm2_metric.column_name()
    model_vectors = self._esm2_metric.load_descriptor_vectors_txt(data[c])

    # compute the RMS of each residue
    rms = np.array([
      np.sqrt(np.mean((self._ref_vectors[i] - model_vectors[i]) ** 2))
      for i in range(self._ref_vectors.shape[0])
    ])

    # then compute the average of the RMS of all residues; this is the same as computing
    # the weighted mean since all residues have the same number of descriptors
    data[self.column_name()] = np.average(rms)
    return data
