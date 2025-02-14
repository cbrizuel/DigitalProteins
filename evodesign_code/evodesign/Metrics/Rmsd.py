from .Metric import Metric
from Bio.PDB import Superimposer
from typing import Optional, List, Dict
from ..Context import Context
from Bio.PDB.Atom import Atom
import pandas as pd





class Rmsd(Metric):

  _superimposer = None

  

  def __init__(self, column: Optional[str] = None) -> None:
    super().__init__(column)
  


  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    """
    Computes the root-mean-squared of the distance between the atom coordinates 
    of a model backbone and a reference backbone after superimposing the former 
    into the latter. Both backbones must contain equal number of atoms.

    This function transforms the coordinates stored in `backbone` so that this
    backbone moves towards the reference backbone (stored in the given context)
    during superposition.

    Parameters
    ----------
    backbone : List[Bio.PDB.Atom.Atom]
        The model backbone. In this case, the model backbone will be moved 
        towards the reference backbone during the superposition.
    data : pandas.Series
        Other data associated with the given backbone. Typically consists of the
        amino acid sequence and other metrics computed from said sequence. 
    context : Context
        The context data used by the calling evolutionary algorithm.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed RMSD value with the current metric's
        column name as the RMSD's key.
    """
    data[self.column_name()] = self._rmsd(backbone, context.ref_backbone)
    return data
  


  def _rmsd(self, 
            backbone: List[Atom], 
            ref_backbone: List[Atom]
            ) -> float:
    if not self._superimposer:
      self._superimposer = Superimposer()
    self._superimposer.set_atoms(ref_backbone, backbone)
    self._superimposer.apply(backbone)
    return self._superimposer.rms
  