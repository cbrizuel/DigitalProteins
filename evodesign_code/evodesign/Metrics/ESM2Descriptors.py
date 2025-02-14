from .Metric import Metric
from ..Workspace import Workspace
import numpy as np
import numpy.typing as npt
import os
from typing import Optional, List, Dict
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd





class ESM2Descriptors(Metric):

  model = None
  batch_converter = None
  


  def _params(self) -> dict:
    params = super()._params()
    params['gpu_device'] = self._gpu_device
    return params
  


  def __init__(self, 
               gpu_device: Optional[str] = "cuda:0",
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    self._gpu_device = gpu_device
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    sequence, sequence_id = data['sequence'], data['sequence_id']
    vectors = self.compute_descriptor_vectors(sequence, sequence_id)
    txt_path = f'{context.workspace.esm2_dir}/{sequence_id}.txt'
    self.save_descriptor_vectors_txt(vectors, txt_path)
    data[self.column_name()] = txt_path
    return data
  


  def compute_descriptor_vectors(self, 
                                 sequence: str,
                                 sequence_id: str = 'prot_0000_0000'
                                 ) -> npt.NDArray[np.float64]:
    # initialize the model if not yet initialized
    import torch
    if self.model is None or self.batch_converter is None:
      import esm
      self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
      self.batch_converter = alphabet.get_batch_converter()
      self.model.eval()
      if torch.cuda.is_available() and self._gpu_device is not None:
        device = torch.device(self._gpu_device)
        self.model = self.model.to(device)
    
    data = [( sequence_id, sequence )]
    seq_ids, seqs, tokens = self.batch_converter(data)
    if torch.cuda.is_available() and self._gpu_device is not None:
      tokens = tokens.to(device='cuda', non_blocking=True)
    with torch.no_grad():
      result = self.model(tokens, 
                          repr_layers=[self.model.num_layers], 
                          return_contacts=False)
    
    # `result['representations']` contains the weights of each layer in the 
    # NN; we only want the weights of the last layer
    last_layer = result['representations'][self.model.num_layers]

    # the last layer contains a certain number of weights per token; a token is
    # an integer representation of each AA in the input sequence, however, 
    # additional tokens are appended at the beginning and at the end of said 
    # sequence; we only want to retrieve the weights corresponding to the AA
    # in the sequence
    matrix = last_layer[0][1 : len(seqs[0]) + 1].cpu().numpy()

    # free GPU memory
    del tokens
    del result

    return matrix
  


  @classmethod
  def save_descriptor_vectors_txt(cls, 
                                  vectors: npt.NDArray[np.float64], 
                                  txt_path: str
                                  ) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    with open(txt_path, 'wt', encoding='utf-8') as txt_file:
      for i in range(vectors.shape[0]):
        vector = vectors[i].tolist()
        txt_file.write(','.join(map(str, vector)) + '\n')

  

  @classmethod
  def load_descriptor_vectors_txt(cls, txt_path: str) -> npt.NDArray[np.float64]:
    vectors = np.array([
      [ float(value.strip()) for value in line.split(',') ]
      for line in open(txt_path, 'rt', encoding='utf-8')
    ])
    return vectors
  