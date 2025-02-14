from .Predictor import Predictor
from typing import List, Tuple
from Bio.PDB.Atom import Atom





class Null(Predictor):
  
    def predict_pdb_str(self, sequence: str) -> str:
        """
        Does nothing.

        Parameters
        ----------
        sequence : str
            Unused.
        """
        pass



    def __call__(self,
                 sequence: str, 
                 pdb_path: str
                 ) -> Tuple[List[Atom], float]:
        """
        Does nothing.

        Parameters
        ----------
        sequence : str
            Unused.
        pdbPath : str
            Unused.

        Returns
        -------
        List[Atom]
            An empty list.
        """
        return [], 0.0
