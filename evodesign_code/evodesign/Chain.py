from typing import List
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom
from Bio.PDB import PDBParser
from typing import Optional
import os



BACKBONE_ATOMS = dict.fromkeys([ 'N', 'CA', 'C', 'O' ])





def load_structure(pdb_path: str, 
                   parser: Optional[PDBParser] = None
                   ) -> Structure:
    """
    Loads a BioPython Structure instance from the specified PDB file.

    Parameters
    ----------
    pdb_path : str
        The path to the PDB file to be opened.

    Returns
    -------
    Bio.PDB.Structure.Structure
        The loaded Structure instance.
    """
    if parser is None: 
        parser = PDBParser()
    structure_id = os.path.splitext(os.path.basename(pdb_path))[0]
    return parser.get_structure(structure_id, pdb_path)



def length(structure: Structure, 
           model_id: int = 0, 
           chain_id: str = 'A'
           ) -> int:
    """
    Returns the number of residues in a given polypeptide chain in the 
    specified BioPython Structure instance.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The BioPython Structure containing the desired chain. Structure 
        objects in BioPython may contain multiple chains organized
        in a hierarchical representation. However, this function can only work
        with a single chain at a time, which must be specified by means of two
        IDs--one for the model and one for the actual chain (see "The Structure
        Object" section provided here: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ).
    model_id : int, optional
        The model ID for finding the desired chain. The default is 0.
    chain_id : str, optional
        The chain ID of the desired chain. The default is 'A'.

    Returns
    -------
    int
        The number of residues in the specified chain.
    """
    return len(list(structure[model_id][chain_id].get_residues()))



def backbone_atoms(structure: Structure,
                   model_id: int = 0, 
                   chain_id: str = 'A'
                   ) -> List[Atom]:
    """
    Returns a list containing the backbone atoms of a certain polypeptide 
    chain in the specified BioPython Structure instance.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The BioPython Structure containing the desired chain. Structure 
        objects in BioPython may contain multiple chains organized
        in a hierarchical representation. However, this function can only work
        with a single chain at a time, which must be specified by means of two
        IDs--one for the model and one for the actual chain (see "The Structure
        Object" section provided here: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ).
    model_id : int, optional
        The model ID for finding the desired chain. The default is 0.
    chain_id : str, optional
        The chain ID of the desired chain. The default is 'A'.

    Returns
    -------
    List[Atom]
        The backbone atoms of the specified chain.
    """
    return [
        atom
        for atom in structure[model_id][chain_id].get_atoms()
        if atom.get_name() in BACKBONE_ATOMS
    ]
