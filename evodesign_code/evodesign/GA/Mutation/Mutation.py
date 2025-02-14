from abc import ABC, abstractmethod
from ...SettingsRetrievable import SettingsRetrievable
import evodesign.Utils as Utils
import numpy as np
import pandas as pd
from typing import Optional, List, Dict





class Mutation(SettingsRetrievable, ABC):

  def __init__(self, mutation_prob: float = 1.0) -> None:
    super().__init__()
    self._mutation_prob = mutation_prob



  def _params(self) -> dict:
    return {
      'mutation_prob': self._mutation_prob
    }



  def __call__(self, 
               rng: np.random.Generator,
               children: pd.DataFrame,
               allowed_letters: Optional[Dict[int, List[str]]] = None
               ) -> pd.DataFrame:
    """
    Modifies the amino acid sequences of a subset of the given population table.

    Parameters
    ----------
    rng : numpy.random.Generator
        The RNG used to mutate the sequence.
    children : pandas.DataFrame
        The table from which a subset will be selected and modified.
    allowed_letters : Dict[int, List[str]], optional
        A description of which letters are allowed to be chosen for certain positions
        in the sequence. If no letter pool is specified for a given position, then no
        restrictions in the letter selection will be imposed at that position. Default
        is `None`, which means that any amino acid letter can be chosen at any position.
    
    Return
    ------
    pandas.DataFrame
        The given population table with some of the sequences mutated.
    """
    indices = children.apply(lambda row: Utils.coin_toss(rng, self._mutation_prob), 
                             axis=1)
    children.loc[indices, 'sequence'] = \
      children.loc[indices, 'sequence'].apply(self.mutate_sequence, 
                                              args=(rng, allowed_letters))
    return children
  


  @abstractmethod
  def mutate_sequence(self, 
                      sequence: str,
                      rng: np.random.Generator,
                      allowed_letters: Optional[Dict[int, List[str]]] = None
                      ) -> str:
    raise NotImplementedError
