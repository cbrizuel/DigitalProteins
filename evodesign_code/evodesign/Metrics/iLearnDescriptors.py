from .Metric import Metric
import subprocess
from ..Workspace import Workspace
from typing import Dict, List, Optional
import numpy.typing as npt
import numpy as np
import os
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd





class iLearnDescriptors(Metric):
  
  def _params(self) -> dict:
    params = super()._params()
    params['ilearn_dir'] = self._ilearn_dir
    params['methods'] = self._methods
    return params



  def __init__(self, 
               ilearn_dir: str,
               methods: List[str] = [
                 'GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC', 'BLOSUM62', 'CTDC',
                 'CTDT', 'CTDD'
               ],
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    self._ilearn_dir = ilearn_dir
    self._methods = methods
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> Dict[str, str]:
    fasta_path = f'{context.workspace.root_dir}/.temp.fasta'
    sequence_id, sequence = data['sequence_id'], data['sequence']
    with open(fasta_path, 'wt', encoding='utf-8') as fasta_file:
      fasta_file.write(f'>{sequence_id}|null|null|null\n{sequence}\n')
    os.remove(fasta_path)
    csv_dir = f'{context.workspace.ilearn_dir}/{sequence_id}'
    vector_paths = self.compute_descriptor_vectors(fasta_path, csv_dir)
    data = data.combine_first(vector_paths)
    data[self.column_name()] = True
    return data
  


  def compute_descriptor_vectors(self, 
                                 fasta_path: str,
                                 csv_dir: str
                                 ) -> pd.Series:
    os.makedirs(csv_dir, exist_ok=True)
    output = {}
    for method in self._methods:
      csv_path = f'{csv_dir}/{method}.csv'
      for line in self._run_ilearn_cmd(method, fasta_path, csv_path):
        print(line, end='')
      output[method] = csv_path
    return pd.Series(output)
  


  @classmethod
  def load_descriptor_vectors(cls, vector_paths: pd.Series) -> Dict[str, npt.NDArray[np.float64]]:
    vectors = {
      method: cls.load_vector_csv(csv_path)
      for method, csv_path in vector_paths.items()
    }
    return vectors
  


  @classmethod
  def load_vector_csv(cls, csv_path: str) -> npt.NDArray[np.float64]:
    for line in open(csv_path, 'rt'):
      values = line.split(',')
    vector = np.array([ float(x.strip()) for x in values if x.find('null') == -1 ])
    return vector



  def _run_ilearn_cmd(self,
                      method: str,
                      input_fasta_path: str,
                      output_csv_path: str
                      ):
    cmd = [
      'python3',
      f'{self._ilearn_dir}/iLearn-protein-basic.py',
      '--method', method,
      '--format', 'csv',
      '--file', input_fasta_path,
      '--out', output_csv_path
    ]
    popen = subprocess.Popen(cmd, 
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
      yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
      raise subprocess.CalledProcessError(return_code, cmd)
  