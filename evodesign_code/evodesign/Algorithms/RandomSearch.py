from .GASimple import GASimple
from ..Fitness.FitnessFunction import FitnessFunction
from ..Metrics.Metric import Metric
from ..Prediction.Predictor import Predictor
import pandas as pd
from typing import List, Optional





class RandomSearch(GASimple):

    def _params(self) -> dict:
        return {
            'max_generations': self._max_generations,
            'population_size': self._population_size,
            'predictor': self._predictor.settings(),
            'metrics': [ metric.settings() for metric in self._metrics ],
            'sort_columns': self._sort_columns,
            'sort_ascending': self._sort_ascending,
            'fitness_fn': self._fitness_fn.settings()
        }



    def __init__(self, 
                 fitness_fn: FitnessFunction,
                 max_generations: int, 
                 population_size: int, 
                 predictor: Predictor,
                 metrics: List[Metric], 
                 sort_columns: Optional[List[str]] = None,
                 sort_ascending: Optional[List[bool]] = None
                 ) -> None:
        super().__init__(fitness_fn,
                         max_generations, 
                         population_size, 
                         predictor, 
                         None, 
                         None, 
                         None, 
                         metrics, 
                         sort_columns, 
                         sort_ascending)



    def next_population(self, population: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the selection, recombination and mutation operators to the given 
        population to create a new population for the next generation of the
        evolutionary algorithm.

        Parameters
        ----------
        population : pandas.DataFrame
            The population from which the new population will be created.

        Returns
        -------
        pandas.DataFrame
            The generated population.
        """
        children = self.initial_population()
        children["generation_id"] = population["generation_id"] + 1
        return children
