from .GenericGA import GenericGA
import evodesign.Statistics as Statistics
import evodesign.Utils as Utils
import pandas as pd





class GASteadyState(GenericGA):

    def compute_statistics(self, population: pd.DataFrame) -> pd.Series:
        """
        Computes the population statistics.

        Parameters
        ----------
        population : pd.DataFrame
            The population which statistics will be computed

        Returns
        -------
        pd.Series
            The top individual in the population, along the computed statistics.
        """
        top_solution = population.iloc[0].copy()
        survivors = population[population['survivor']]
        top_solution['sequence_identity'] = \
            Statistics.average_sequence_identity(survivors)
        top_solution['lost_amino_acids'] = \
            Statistics.average_amino_acid_loss(survivors)
        return top_solution



    def replacement(self,
                    population: pd.DataFrame,
                    children: pd.DataFrame
                    ) -> pd.DataFrame:
        """
        Creates a new population by selecting which individuals from last 
        generation's population and current generation's children should survive
        to the next generation.

        Parameters
        ----------
        population : pd.DataFrame
            The population from last generation.
        children : pd.DataFrame
            The children created at the current generation.

        Returns
        -------
        pd.DataFrame
            The population for next generation, composed of the selected survivors.
        """
        children.sort_values(by=self._sort_columns,
                             ascending=self._sort_ascending,
                             inplace=True,
                             ignore_index=True)
        last_pop = population[population['survivor']].copy()
        last_pop['survivor'] = False
        last_pop['generation_id'] = children.iloc[0]['generation_id']
        mixed = Utils.merge(last_pop,
                            children,
                            self._sort_columns,
                            self._sort_ascending)
        mixed.loc[:self._population_size - 1, 'survivor'] = True
        return mixed
