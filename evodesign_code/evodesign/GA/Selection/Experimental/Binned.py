from ..Selection import Selection
from ....Context import Context
import pandas as pd
from typing import Tuple





class Binned(Selection):
  
    def _params(self) -> dict:
        params = super()._params()
        params['upper_size'] = self._upper_size
        params['upper_prob'] = self._upper_prob
        params['lower_prob'] = self._lower_prob
        return params
    
  
  
    def __init__(self, 
                 upper_size: int,
                 upper_prob: float = 0.8,
                 lower_prob: float = 0.2
                 ) -> None:
        """
        Selection operator where the population is divided into to groups, which
        we call the "upper bin" and "lower bin". The upper bin contains the top
        `upper_size` individuals according to their fitness, and the lower bin
        contains all remaining individuals. Then, individuals are selected in pairs,
        where each pair could be formed by two randomly selected individuals from
        the upper bin, two from the lower bin, or one from each bin. Which of 
        these three options will be taken for a given pair is chosen randomly. 
        All individuals are selected from their bins with a uniform distribution and
        without replacement.
    
        Parameters
        ----------
        upper_size : int
            The number of individuals in the upper bin.
        upper_prob : float, optional
            The probability for selecting a pair of individuals from the upper bin.
            The default is 0.8.
        lower_prob : float, optional
            The probability for selecting a pair of individuals from the lower bin.
            The default is 0.2. The probability for selecting  mixed pair of 
            individuals is always 1.0 - upper_prob - lower_prob.
        """
        super().__init__()
        self._upper_size = upper_size
        self._upper_prob = upper_prob
        self._lower_prob = lower_prob
        self._weights = [
            upper_prob,
            lower_prob,
            1.0 - upper_prob - lower_prob
        ]
    
  
  
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
        upper_bin = population.iloc[:self._upper_size]
        lower_bin = population.iloc[self._upper_size:]
        option = context.rng.choice([ 0, 1, 2 ], p=self._weights)
        if option == 0:
            selection = context.rng.choice(upper_bin.index, size=2, replace=False)
            parents = ( upper_bin.iloc[selection[0]], upper_bin.iloc[selection[1]] )
        if option == 1:
            selection = context.rng.choice(lower_bin.index, size=2, replace=False)
            parents = ( lower_bin.iloc[selection[0]], lower_bin.iloc[selection[1]] )
        if option == 2:
            m = context.rng.choice(upper_bin.index)
            f = context.rng.choice(lower_bin.index)
            parents = ( population.iloc[m], population.iloc[f] )
        return parents
