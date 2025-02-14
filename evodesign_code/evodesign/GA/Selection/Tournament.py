from .Selection import Selection
from ...Context import Context
import pandas as pd
from typing import List, Tuple





class Tournament(Selection):
  
    def _params(self) -> dict:
        params = super()._params()
        params['tournament_size'] = self._tournament_size
        return params
    
    
    
    def __init__(self, 
                 tournament_size: int
                 ) -> None:
        """
        Selection operator in which a random uniform sample of size 
        `tournament_size`, without replacement, is taken from the population, and 
        the individual with higher fitness from this sample is then selected. 
    
        Notice that it is possible for the same individual to be chosen multiple
        times. However, it is guaranteed that each consecutive pairs of 
        individuals will be distinct.
    
        Parameters
        ----------
        tournament_size : int
            The number of individuals to be randomly chosen to participate in 
            a tournament. Only one of these individuals will be chosen.
        """
        super().__init__()
        self._tournament_size = tournament_size
  
  
  
    def select_parent_couple(self, 
                             population: pd.DataFrame,
                             context: Context
                             ) -> Tuple[pd.Series]:
        """
        Selects a subset of individuals from the given population.
    
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
        mother = self.tournament_selection(population, context)
        father = self.tournament_selection(population, context)
        return ( mother, father )
    
  
  
    def tournament_selection(self, 
                             population: pd.DataFrame,
                             context: Context
                             ) -> pd.Series:
        selection = context.rng.choice(population.index, 
                                       size=self._tournament_size, 
                                       replace=False)
        tournament = population.loc[selection]
        tournament.sort_values(by=context.sort_columns, 
                               ascending=context.sort_ascending,
                               inplace=True, 
                               ignore_index=True)
        return tournament.iloc[0]
    