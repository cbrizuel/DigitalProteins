from ..Metric import Metric
from ..iLearnDescriptors import iLearnDescriptors
from ..ESM2Descriptors import ESM2Descriptors
from ...Context import Context
from typing import List, Optional
import numpy as np
import numpy.typing as npt
import scipy.stats as sp
import pandas as pd
from Bio.PDB.Atom import Atom





class Descriptors(Metric):

  _DESCRIPTOR_METHOD_TYPES = {
    'GAAC': 'probabilistic',
    'EGAAC': 'non_probabilistic', 
    'CKSAAGP': 'non_probabilistic',
    'GDPC': 'probabilistic',
    'GTPC': 'probabilistic',
    'BLOSUM62': 'non_probabilistic',
    'CTDC': 'non_probabilistic',
    'CTDT': 'non_probabilistic',
    'CTDD': 'non_probabilistic'
  }
  


  def _params(self) -> dict:
    params = self._params()
    params['weight'] = self._weight
    params['ilearm_metric'] = self._ilearn_metric.settings()
    params['esm2_metric'] = self._esm2_metric.settings()
    return params
  


  def __init__(self,
               weight: float,
               ilearn_metric: iLearnDescriptors,
               esm2_metric: Optional[ESM2Descriptors] = None,
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    self._weight = weight
    self._ilearn_metric = ilearn_metric
    if esm2_metric is None:
      esm2_metric = ESM2Descriptors()
    self._esm2_metric = esm2_metric
    self._ref_vectors = None
    self._method_types = self._DESCRIPTOR_METHOD_TYPES.copy()
    self._method_types[self._esm2_metric.column_name()] = 'non_probabilistic'
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    # TODO cargar los descriptores de los archivos almacenados si estos ya existen
    # compute the reference vectors if not already done
    if self._ref_vectors is None:
      # compute the iLearn descriptors
      vector_paths = self._ilearn_metric.compute_descriptor_vectors(
        context.workspace.target_fasta_path,
        f'{context.workspace.ilearn_dir}/reference')
      self._ref_vectors = self._ilearn_metric.load_descriptor_vectors(vector_paths)

      # compute the ESM2 descriptors
      esm2_vectors = self._esm2_metric.compute_descriptor_vectors(context.ref_sequence)
      txt_path = f'{context.workspace.esm2_dir}/reference.txt'
      self._esm2_metric.save_descriptor_vectors_txt(esm2_vectors, txt_path)

      # collapse ESM2 vectors into a single vector
      self._ref_vectors[self._esm2_metric.column_name()] = np.concatenate(esm2_vectors, 
                                                                          axis=0)
    
    # compute the iLearn descriptors for the model
    data = self._ilearn_metric(backbone, data, context)
    vector_paths = data[self._ilearn_metric._methods]
    model_vectors = self._ilearn_metric.load_descriptor_vectors(vector_paths)

    # compute the ESM2 descriptors for the model
    data = self._esm2_metric(backbone, data, context)
    c = self._esm2_metric.column_name()
    esm2_vectors = self._esm2_metric.load_descriptor_vectors_txt(data[c])

    # collapse all vectors into a single vector
    model_vectors[c] = np.concatenate(esm2_vectors, axis=0)

    # compute the divergence score
    divergence_scores = np.array([
      self._symmetric_kullback_leibler(self._ref_vectors[method], 
                                       model_vectors[method], 
                                       self._method_types[method])
      for method in self._ilearn_metric._methods + [ self._esm2_metric.column_name() ]
    ])
    score = 2 * self._weight / (1 + divergence_scores.mean())
    data[self.column_name()] = score
    return data
  


  def _symmetric_kullback_leibler(self,
                                  reference: npt.NDArray[np.float64],
                                  model: npt.NDArray[np.float64],
                                  method_type: str
                                  ) -> float:
    a = self._kullback_leibler(reference, model, method_type)
    b = self._kullback_leibler(model, reference, method_type)
    return a + b



  def _kullback_leibler(self,
                        reference: npt.NDArray[np.float64],
                        model: npt.NDArray[np.float64],
                        method_type: str
                        ) -> float:
    a, b = reference, model
    if method_type == 'probabilistic':
      # we do this so that the probability is non-zero
      a += 0.01
      b += 0.01
    else:
      min_value = min([ reference.min(), model.min() ])
      max_value = max([ reference.max(), model.max() ])
      dist_range = (min_value, max_value)
      a, _ = np.histogram(reference, 
                          bins=20, 
                          range=dist_range, 
                          density=False)
      b, _ = np.histogram(model, 
                          bins=20, 
                          range=dist_range,
                          density=False)
      # numpy does not like mixing int with float
      if a.dtype != np.float64: a = a.astype(np.float64)
      if b.dtype != np.float64: b = b.astype(np.float64)
      # we do this so that the probability is non-zero
      a += 0.5
      b += 0.5
    divergence = sp.entropy(a, b)
    return divergence
