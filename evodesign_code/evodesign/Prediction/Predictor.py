from abc import ABC, abstractmethod
from typing import List, Tuple
from Bio.PDB.Atom import Atom
import evodesign.Chain as Chain
from ..SettingsRetrievable import SettingsRetrievable
import numpy as np
import os





class Predictor(SettingsRetrievable, ABC):

    @abstractmethod
    def predict_pdb_str(self, sequence: str) -> str:
        raise NotImplementedError
  


    def predict_pdb_file(self, 
                         sequence: str, 
                         pdb_path: str
                         ) -> None:
        """
        Predicts the 3D structure of a given amino acid sequence using the 
        current model. The resulting prediction is stored in a PDB file in
        the specified file path.

        Parameters
        ----------
        sequence : str
            The amino acid sequence which structure will be predicted. Each residue
            must be represented with a single letter corresponding to one of the
            20 essential amino acids.
        pdb_path : str
            The path and name of the PDB file where the predicted structure will
            be stored.
        """
        if not os.path.isfile(pdb_path):
            os.makedirs(os.path.dirname(os.path.abspath(pdb_path)), exist_ok=True)
        prediction = self.predict_pdb_str(sequence)
        with open(pdb_path, 'wt', encoding='utf-8') as pdb_file:
            pdb_file.write(prediction) 
  


    def __call__(self,
                 sequence: str, 
                 pdb_path: str
                 ) -> Tuple[List[Atom], float]:
        # PDBParser cannot produce an instance of `Structure` directly from
        # a raw PDB string, it can only do it by reading from a PDB file.
        # Thus, the predicted structure must be stored first, before 
        # loading the `Structure` instance.
        self.predict_pdb_file(sequence, pdb_path)
        structure = Chain.load_structure(pdb_path)
        bfactors = np.array([
            atom.get_bfactor() / 100.0 
            if atom.get_bfactor() > 1.0 
            else atom.get_bfactor()
            for atom in structure.get_atoms()
        ])
        return Chain.backbone_atoms(structure), bfactors.mean()
