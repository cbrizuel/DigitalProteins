from evodesign.Context import Context
from .GASimpleElitism import GASimpleElitism
from ..GA.Mutation.Mutation import Mutation
from ..GA.Recombination.Recombination import Recombination
from ..GA.Selection.Selection import Selection
from ..Prediction.Predictor import Predictor
from ..Fitness.FitnessFunction import FitnessFunction
from typing import Optional, List
from ..Metrics.Metric import Metric
import evodesign.Statistics as Statistics
import pandas as pd
import evodesign.Utils as Utils





class GASimple(GASimpleElitism):

    def __init__(self, 
                 fitness_fn: FitnessFunction,
                 max_generations: int, 
                 population_size: int, 
                 predictor: Predictor, 
                 selection: Selection, 
                 recombination: Recombination, 
                 mutation: Mutation,
                 metrics: List[Metric],
                 sort_columns: Optional[List[str]] = None,
                 sort_ascending: Optional[List[bool]] = None
                 ) -> None:
        super().__init__(fitness_fn,
                         max_generations, 
                         population_size,
                         predictor,
                         selection,
                         recombination, 
                         mutation,
                         metrics,
                         sort_columns,
                         sort_ascending)
        self._top_solution = None
    


    def setup(self, 
              context: Context, 
              workspace_root: str, 
              rng_seed: Optional[int] = None
              ) -> None:
        super().setup(context, workspace_root, rng_seed)
        self._top_solution = None
    


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
        if self._top_solution is None:
            self._top_solution = Utils.find_top_solution(population, 
                                                         self._sort_columns, 
                                                         self._sort_ascending)
        top_solution = self._top_solution.copy()
        survivors = population[population['survivor']]
        sample = survivors.sample(30, replace=False)
        top_solution['sequence_identity'] = \
            Statistics.average_sequence_identity(sample)
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
        children["survivor"] = True
        top_offspring = Utils.find_top_solution(children, 
                                                self._sort_columns,
                                                self._sort_ascending)
        if self._top_solution is None:
            self._top_solution = top_offspring
            return children
        self._top_solution["generation_id"] = top_offspring["generation_id"]
        if Utils.is_sorted_before(top_offspring, 
                                  self._top_solution, 
                                  self._sort_columns, 
                                  self._sort_ascending):
            self._top_solution = top_offspring
        return children
