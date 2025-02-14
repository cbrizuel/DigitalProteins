from .Recombination import Recombination
from typing import Tuple
import numpy as np





"""
Creates new sequences by splitting both parent sequences exactly in half  
and mixing the half from one parent with the complementary half from 
the other.

For example, given the sequences 'AAAAAA' and 'DDDDDD', this operator would
produce the sequences 'AAADDD' and 'DDDAAA'.
"""
class MiddlePointCrossover(Recombination):

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
    n = len(mother)
    i = n / 2 if n % 2 == 0 else (n - 1) / 2
    sister = mother[0:i] + father[i:]
    brother = father[0:i] + mother[i:]
    return ( sister, brother )
