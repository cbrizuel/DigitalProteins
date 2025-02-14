from abc import ABC, abstractmethod
from ...SettingsRetrievable import SettingsRetrievable
from ...Context import Context
import evodesign.Utils as Utils
import pandas as pd
from typing import Tuple





class Selection(SettingsRetrievable, ABC):

    def __call__(self,
                 population: pd.DataFrame,
                 context: Context
                 ) -> pd.DataFrame:
        """
        Selects a subset of individuals from the given population. Only those
        rows with a `True` value in the 'survivor' column are considered.
    
        Parameters
        ----------
        population : pandas.DataFrame
            The population to be sampled.
        context : Context
            The context data used by the calling evolutionary algorithm.
    
        Returns
        -------
        pandas.DataFrame
            The selected subset of individuals.
        """
        survivors = population[population['survivor']]
        parents = pd.DataFrame(columns=survivors.columns)
        for _ in range(len(survivors)):
            selection = self.select_parent_couple(population, context)
            parents = Utils.df_append(parents, selection[0])
            parents = Utils.df_append(parents, selection[1])
        return parents
  


    @abstractmethod
    def select_parent_couple(self, 
                             population: pd.DataFrame,
                             context: Context
                             ) -> Tuple[pd.Series]:
        raise NotImplementedError
