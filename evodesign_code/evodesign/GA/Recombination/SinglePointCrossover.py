from .Recombination import Recombination
from typing import Tuple
import numpy as np





"""
Creates new sequences by splitting the parent sequences at a randomly 
chosen residue position and mixing one part from one of the parents with
the complementary part from the other.

For example, given the sequences 'AAAAAA' and 'DDDDDD', and assuming that 
it was chosen to split these sequences at the 4th position, then this 
operator would produce the sequences 'AAAADD' and 'DDDDAA'.
"""
class SinglePointCrossover(Recombination):
  
  def offspring_sequences(self, 
                          rng: np.random.Generator,
                          mother: str,
                          father: str
                          ) -> Tuple[str]:
    """
    Mix partial information from the two given sequences to create two new
    sequences. It is assumed that both sequences have equal length.
    
    In both sequences, each residue must be represented by a single letter
    corresponding to one of the 20 essential amino acids.

    Parameters
    ----------
    rng : numpy.random.Generator
        The pseudo-random number generator.
    mother : str
        One of the sequences to be mixed.
    father : str
        The other sequence to be mixed.

    Returns
    -------
    List[str]
        The two sequences produced.
    """
    i = rng.integers(0, len(mother))
    sister = mother[0:i] + father[i:]
    brother = father[0:i] + mother[i:]
    return ( sister, brother )
