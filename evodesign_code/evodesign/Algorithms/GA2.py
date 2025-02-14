from .GenericGA import GenericGA
from typing import Optional, List
from ..Prediction.Predictor import Predictor
from ..Fitness.FitnessFunction import FitnessFunction
from ..GA.Selection.Selection import Selection
from ..GA.Recombination.Recombination import Recombination
from ..GA.Mutation.Mutation import Mutation
import evodesign.Statistics as Statistics
import evodesign.Utils as Utils
import pandas as pd





# TODO refactorizar para implementar context
class GA2(GenericGA):
  
  def _params(self) -> dict:
    params = super()._params()
    params['elitismSize'] = self._elitism_size
    params['betterOffspringBias'] = self._better_bias
    return params
  


  def __init__(self,
               maxGenerations: int,
               popSize: int,
               elitismSize: int,
               predictor: Predictor,
               fitnessFn: FitnessFunction,
               selection: Selection,
               recombination: Recombination,
               mutation: Mutation,
               betterOffspringBias: float = 1.0,
               sortColumns: Optional[List[str]] = None,
               sortAscending: Optional[List[bool]] = None
               ) -> None:
    """
    A genetic algorithm which optimizes some fitness function obtained from
    comparing the target backbone with the model backbone, which is predicted
    using a protein structure prediction algorithm like ESMFold or AlphaFold.

    Parameters
    ----------
    maxGenerations : int
        The maximum number of generations for which the algorithm will be 
        executed.
    popSize : int
        The number of sequences in the population in each generation.
    elitismSize : int
        The number of sequences with the highest fitness value in the population
        that will be treated as the 'elite' population.
    betterOffspringBias : float
        The probability for selecting the offspring with better fitness for 
        survival to the next generation. All the recombination operations that
        can work with this algorithm are assumed to always produce two children,
        for each pair of parent sequences, but only one will be considered.
        The default value is 1.0.
    predictor : Predictor
        The protein structure prediction algorithm that will be used.
    fitnessFn : FitnessFunction
        The fitness funtion that will be optimized.
    selection : Selection
        The parents selection operation that will be used.
    recombination : Recombination
        The parents recombination operation that will be used.
    mutation : Mutation
        The sequence mutation operation that will be used.
    """
    super().__init__(maxGenerations,
                     popSize,
                     predictor,
                     fitnessFn,
                     selection,
                     recombination,
                     mutation,
                     sortColumns,
                     sortAscending)
    self._elitism_size = elitismSize
    self._better_bias = betterOffspringBias
    self._offspring_selection = _OffspringSelection(fitnessFn.column_name(), 
                                                    betterOffspringBias)
  


  def compute_statistics(self, population: pd.DataFrame) -> pd.Series:
    """
    Computes the average number of missing amino acids and the average sequence 
    identity amongst the sequences in the upper bin in the given population.
    Only the upper bin is considered because, if its diversity is lost, then 
    the algorithm would've converged in a local optimum.

    Parameters
    ----------
    population : pandas.DataFrame
        The population from which the statistics will be computed.

    Returns
    -------
    pandas.Series
        The information of the top solution in the given population, alongside
        the computed statistics for said population.
    """
    top_solution = population.iloc[0].copy()
    upper_bin = population.iloc[:self._elitism_size]
    top_solution['sequence_identity'] = \
      Statistics.average_sequence_identity(upper_bin)
    top_solution['lost_amino_acids'] = \
      Statistics.average_amino_acid_loss(upper_bin)
    return top_solution
  


  def replacement(self, 
                  population: pd.DataFrame, 
                  children: pd.DataFrame
                  ) -> pd.DataFrame:
    # TODO documentar esta función y hacer sus pruebas unitarias
    children.sort_values(by=self._sort_cols, 
                         ascending=self._sort_ascending, 
                         inplace=True,
                         ignore_index=True)
    children = self._offspring_selection(children)
    last_upper_bin = population.iloc[:self._elitism_size].copy()
    upper_bin = children.iloc[:self._elitism_size]
    last_upper_bin['generation_id'] = children.iloc[0]['generation_id']
    upper_bin = Utils.merge(upper_bin, 
                            last_upper_bin, 
                            self._sort_cols, 
                            self._sort_ascending)
    survivors_upper = upper_bin.iloc[:self._elitism_size]
    dead_upper = upper_bin.iloc[self._elitism_size:].copy()
    dead_upper['survivor'] = False
    objs = [ 
      survivors_upper, 
      children.iloc[self._elitism_size:],
      dead_upper
    ]
    results = pd.concat(objs, axis=0, ignore_index=True)
    return results





class _OffspringSelection:
  # TODO documentar esta clase y hacer sus pruebas unitarias

  def __init__(self,
               fitnessColumn: str,
               betterFitnessBias: float = 1.0
               ) -> None:
    self._fitness_col = fitnessColumn
    self._better_bias = betterFitnessBias
  


  def __call__(self, children: pd.DataFrame) -> pd.DataFrame:
    # we assume that the recombination operation used produced two children
    # per parent pair; thus, if the number of children is odd, then disregard
    # the last element
    if len(children) % 2 != 0:
      children = children.iloc[:-1]
    survivors, dead = pd.DataFrame(), pd.DataFrame()
    for i in range(0, len(children), 2):
      better = children.iloc[i][self._fitness_col]
      worse = children.iloc[i + 1][self._fitness_col]
      b, w = (i, i + 1) if better >= worse else (i + 1, i)
      s, d = (b, w) if Utils.coin_toss(self._better_bias) else (w, b)
      selected_child = children.iloc[s].copy()
      selected_child['survivor'] = True
      survivors = pd.concat([ survivors, selected_child.to_frame().T ],
                            ignore_index=True)
      dead = pd.concat([ dead, children.iloc[d].to_frame().T ],
                       ignore_index=True)
    results = pd.concat([ survivors, dead ], ignore_index=True)
    return results
