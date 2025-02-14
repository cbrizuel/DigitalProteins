from .Algorithm import Algorithm
from abc import ABC
from typing import Optional, List
from ..GA.Mutation.Mutation import Mutation
from ..GA.Recombination.Recombination import Recombination
from ..GA.Selection.Selection import Selection
from ..Prediction.Predictor import Predictor
from ..Fitness.FitnessFunction import FitnessFunction
import evodesign.Sequence as Sequence
import pandas as pd
import matplotlib.pyplot as plt
import os
from ..Metrics.Metric import Metric





class GenericGA(Algorithm, ABC):
  
  def _params(self) -> dict:
    params = super()._params()
    params['sort_columns'] = self._sort_columns
    params['sort_ascending'] = self._sort_ascending
    params['fitness_fn'] = self._fitness_fn.settings()
    return params
  


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
    if not sort_columns:
      # if two individuals are tied in fitness, break the tie using the plddt
      sort_columns = [ fitness_fn.column_name(), 'plddt' ]
    if not sort_ascending:
      # by default sort in descending order; higher fitness comes at the top
      sort_ascending = len(sort_columns) * [ False ]
    super().__init__(max_generations,
                     population_size,
                     predictor,
                     selection,
                     recombination,
                     mutation,
                     metrics,
                     sort_columns,
                     sort_ascending)
    self._fitness_fn = fitness_fn



  def save_statistics_graph(self, stats: pd.DataFrame) -> None:
    """
    Creates a graph for the given statistics and saves it as a PNG file in the
    workspace.

    Parameters
    ----------
    stats : pandas.DataFrame
        The statistics which data will be graphed and stored.
    """
    os.makedirs(self._context.workspace.root_dir, exist_ok=True)
    plt.ioff() # turn off the GUI
    fig, ax = plt.subplots(ncols=2, figsize=(14, 6))
    fig.suptitle(self._context.workspace.root_dir)
    c = self._fitness_fn.column_name()
    series1 = ax[0].plot(stats['generation_id'], stats[c], color='C0', label=c)
    ax[0].tick_params(axis='y', labelcolor='C0')
    ax[0].set_xlabel('generation_id')
    ax1 = ax[0].twinx()
    series2 = ax1.plot(stats['generation_id'], 
                       stats['plddt'], 
                       color='C1', 
                       label='plddt')
    ax1.tick_params(axis='y', labelcolor='C1')
    series = series1 + series2
    labels = [ x.get_label() for x in series ]
    ax[0].legend(series, labels, loc='best')
    series1 = ax[1].plot(stats['generation_id'], 
                         stats['sequence_identity'], 
                         color='C2',
                         label='sequence_identity')
    ax[1].tick_params(axis='y', labelcolor='C2')
    sequence_length = len(stats.iloc[0]['sequence'])
    ax[1].set_ylim(bottom=0., top=sequence_length)
    ax[1].set_xlabel('generation_id')
    ax2 = ax[1].twinx()
    series2 = ax2.plot(stats['generation_id'], 
                       stats['lost_amino_acids'], 
                       color='C3',
                       label='lost_amino_acids')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.set_ylim(bottom=0, top=len(Sequence.AMINO_ACIDS) - 1)
    series = series1 + series2
    labels = [ x.get_label() for x in series ]
    ax[1].legend(series, labels, loc='best')
    plt.savefig(f'{self._context.workspace.root_dir}/fitness_diversity.png')
    plt.close()
  


  def compute_population_fitness(self, 
                                 population: pd.DataFrame,
                                 temporary: bool = False
                                 ) -> pd.DataFrame:
    """
    Computes the fitness for each sequence in the given population.

    Parameters
    ----------
    population : pandas.DataFrame
        The population for which sequences the fitness will be computed.
    temporary : bool, optional
        Indicates if the data of the current population data should be stored in 
        the temporary storage location or the permanent one in case an exception
        is raised during the fitness computation process.

    Returns
    -------
    pandas.DataFrame
        The original population with added columns to include the computed 
        metrics and fitness values.
    """
    subset = population
    c = self._fitness_fn.column_name()
    if c in population.columns:
      indices = population[c].isna()
      subset = population[indices]
    
    # we could use DataFrame.apply, but then we wouldn't be able to catch 
    # exceptions and save the progress mid-loop
    try:
      for idx, row in subset.iterrows():
        pdb_path = f'{self._context.workspace.pdbs_dir}/{row["sequence_id"]}.pdb'
        backbone, row['plddt'] = self._predictor(row['sequence'], pdb_path)
        # first, compute all metrics
        for metric in self._metrics:
          row = metric(backbone, row, self._context)
        # then, compute the fitness function proper
        row = self._fitness_fn(row)
        for column, value in row.items():
          population.at[idx, column] = value
    except BaseException as e:
      self._context.workspace.save_population(population, temporary)
      raise e
    return population
  


  def compute_initial_fitness(self, population: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the fitness of the population of the first generation, and sorts
    the individuals according to their fitness in descending order. If the
    given population has the fitness values already computed for some of the
    individuals, only the individuals with missing fitness values will be
    processed.

    Parameters
    ----------
    population : pandas.DataFrame
        The population for which individuals the fitness value will be computed.
    reference : List[Bio.PDB.Atom.Atom]
        The backbone of the target protein. The fitness of each individual
        in the population can be computed from this backbone.
    refSequence: str, optional
        The amino acid sequence of the target protein. The fitness of each
        individual in the population can be computed from this sequence.

    Returns
    -------
    pandas.DataFrame
        The population updated with the fitness values for all individuals.
    """
    c = self._fitness_fn.column_name()
    if c not in population.columns or population[c].isna().sum() > 0:
      population = self.compute_population_fitness(population)
      population.sort_values(by=self._sort_columns, 
                             ascending=self._sort_ascending,
                             inplace=True,
                             ignore_index=True)
    return population
  


  def termination(self, population: pd.DataFrame) -> bool:
    """
    Checks if the best solution in the given population has reached the upper 
    bound for the fitness value.

    Parameters
    ----------
    population : pandas.DataFrame
        The population for which the termination condition will be checked.

    Returns
    -------
    bool
        A flag that indicates if the fitness upper bound has been reached.
    """
    f = self._fitness_fn
    return population.iloc[0][f.column_name()] >= f.upper_bound
  