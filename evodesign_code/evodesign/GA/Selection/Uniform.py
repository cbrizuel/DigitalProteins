from .Selection import Selection
from ...Context import Context
import pandas as pd
from typing import Tuple





class Uniform(Selection):

    def select_parent_couple(self, 
                             population: pd.DataFrame,
                             context: Context
                             ) -> Tuple[pd.Series]:
        """
        Selects a couple of individuals from the given population.
    
        Parameters
        ----------
        population : pandas.DataFrame
            The population to be sampled.
        context : Context
            The context data used by the calling evolutionary algorithm.
    
        Returns
        -------
        List[pandas.DataFrame]
            The selected individuals.
        """
        i = context.rng.choice(population.index, size=2, replace=True)
        selection = population.iloc[i]
        return ( selection.iloc[0], selection.iloc[1] )
