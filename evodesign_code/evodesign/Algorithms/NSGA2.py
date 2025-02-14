from .Algorithm import Algorithm
from typing import List, Tuple, Optional
from ..Prediction.Predictor import Predictor
from ..Fitness.FitnessFunction import FitnessFunction
from ..GA.Recombination.Recombination import Recombination
from ..GA.Mutation.Mutation import Mutation
from ..GA.Selection.Tournament import Tournament
import evodesign.Statistics as Statistics
import evodesign.Sequence as Sequence
import evodesign.Utils as Utils
from Bio.PDB.Atom import Atom
import pandas as pd
import matplotlib.pyplot as plt
import math
import os




# TODO refactorizar para implementar context
class NSGA2(Algorithm):

  def _params(self) -> dict:
    params = super()._params()
    params['fitnessFns'] = [ f.settings() for f in self._fitness_fns ]
    params['tournamentElitism'] = self._selection._params()['elitism']
    del params['selection']
    return params
  


  def __init__(self, 
               maxGenerations: int,
               popSize: int,
               predictor: Predictor,
               fitnessFns: List[FitnessFunction],
               recombination: Recombination,
               mutation: Mutation,
               tournamentElitism: bool = True
               ) -> None:
    super().__init__(maxGenerations,
                     popSize,
                     predictor,
                     Tournament(tournamentSize=2, 
                                sortColumns=[ 'rank', 'distance' ],
                                sortAscending=[ True, False ],
                                elitism=tournamentElitism),
                     recombination,
                     mutation)
    self._fitness_fns = fitnessFns
    self._elitism = tournamentElitism
  


  def compute_statistics(self, population: pd.DataFrame) -> pd.Series:
    survivors = population[population['survivor']]
    data = {
      'generation_id': population.iloc[0]['generation_id'],
      'non_dominated_front_size': len(survivors[survivors['rank'] == 1]),
      'sequence_identity': Statistics.average_sequence_identity(survivors),
      'lost_amino_acids': Statistics.average_amino_acid_loss(survivors),
      'sequence_length': len(survivors.iloc[0]['sequence'])
    }
    return pd.Series(data)
  


  def save_statistics_graph(self, stats: pd.DataFrame) -> None:
    os.makedirs(self._context.workspace.root_dir, exist_ok=True)
    plt.ioff() # turn off the GUI
    fig, ax1 = plt.subplots()
    series1 = ax1.plot(stats['generation_id'], 
                       stats['sequence_identity'], 
                       color='C2',
                       label='sequence_identity')
    ax1.tick_params(axis='y', labelcolor='C2')
    ax1.set_ylim(bottom=0., top=stats.iloc[-1]['sequence_length'])
    ax1.set_xlabel('generation_id')
    ax2 = ax1.twinx()
    series2 = ax2.plot(stats['generation_id'], 
                       stats['lost_amino_acids'], 
                       color='C3',
                       label='lost_amino_acids')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.set_ylim(bottom=0, top=len(Sequence.AMINO_ACIDS))
    series = series1 + series2
    labels = [ x.get_label() for x in series ]
    ax1.legend(series, labels, loc='best')
    plt.savefig(f'{self._context.workspace.root_dir}/fitness_diversity.png')
    plt.close()



  def compute_fitness(self, 
                      population: pd.DataFrame,
                      reference: List[Atom],
                      refSequence: Optional[str] = None,
                      **kwargs
                      ) -> pd.DataFrame:
    # for multiobjective optimization, either all the fitness values are 
    # present in the population or none of them are present
    subset = population
    if self._fitness_fns[0].column_name() in population.columns:
      indices = population[self._fitness_fns[0].column_name()].isna()
      subset = population[indices]
    
    try:
      for idx, row in subset.iterrows():
        filename = f'{self._context.workspace.pdbs_dir}/{row["sequence_id"]}.pdb'
        model, row['plddt'] = self._predictor(row['sequence'], filename)
        for f in self._fitness_fns:
          results = f(model=model, 
                      reference=reference, 
                      refSequence=refSequence,
                      sequence=row['sequence'],
                      sequenceId=row['sequence_id'],
                      plddt=row['plddt'])
          row = row.combine_first(results)
        for column, value in row.items():
          population.at[idx, column] = value
    except BaseException as e:
      self._context.workspace.save_population(population, **kwargs)
      raise e
    return population
  


  def initial_fitness_computation(self, 
                                  population: pd.DataFrame,
                                  reference: List[Atom],
                                  refSequence: Optional[str] = None,
                                  ) -> pd.DataFrame:
    # we assume that either all the fitness values are present in the DataFrame
    # or none of them are
    c = self._fitness_fns[0].column_name()
    if c not in population.columns or population[c].isna().sum() > 0:
      population = self.compute_fitness(population, reference, refSequence)
      # sort the population by rank in ascending order
      population, fronts = self.non_domination_rank(population)
      population = pd.DataFrame(columns=population.columns)
      for front in fronts:
        front = self.compute_crowding_distance(front)
        population = pd.concat([ population, front ], ignore_index=True)
    return population
  


  def replacement(self, 
                  population: pd.DataFrame, 
                  children: pd.DataFrame
                  ) -> pd.DataFrame:
    mixed = pd.concat([ population[population['survivor']], children ], 
                      ignore_index=True)
    mixed['survivor'] = False
    mixed['generation_id'] = children.iloc[0]['generation_id']
    mixed, fronts = self.non_domination_rank(mixed)
    # we create the next population by filling it row by row
    next_pop = pd.DataFrame(columns=mixed.columns)
    i, front = 0, fronts[0]
    while len(next_pop) + len(front) < self._population_size:
      front = self.compute_crowding_distance(front)
      next_pop = pd.concat([ next_pop, front ], ignore_index=True)
      i += 1
      front = fronts[i]
    # at this point, there are more sequences in the front `i` than those that
    # fit in the new population, so we sort by rank and crowding distance and
    # preserve the top sequences that fit
    front = self.compute_crowding_distance(front)
    front.sort_values(by=[ 'rank', 'distance' ],
                      ascending=[ True, False ],
                      inplace=True,
                      ignore_index=True)
    n = self._population_size - len(next_pop)
    next_pop = pd.concat([ next_pop, front.iloc[:n] ], ignore_index=True)

    # finish by marking all these sequences as survivors and appending the rest
    next_pop['survivor'] = True
    next_pop = pd.concat([ next_pop, front[n:] ], ignore_index=True)
    for j in range(i, len(fronts)):
      next_pop = pd.concat([ next_pop, fronts[j] ], ignore_index=True)
    return next_pop
  

  
  def compute_crowding_distance(self, front: pd.DataFrame) -> pd.DataFrame:
    # TODO escribir las pruebas unitarias de esta función
    front['distance'] = 0.0
    for f in self._fitness_fns:
      c = f.column_name()
      df = front.sort_values(by=c, ignore_index=True)
      df.at[0, 'distance'] = df.at[len(df) - 1, 'distance'] = math.inf
      denom = df.iloc[0][c] - df.iloc[-1][c]
      for i in range(2, len(df) - 1):
        a, b = df.iloc[i + 1][c], df.iloc[i - 1][c]
        df.at[i, 'distance'] += abs((a - b) / denom)
      front = df
    return front
  


  def non_domination_rank(self, 
                          population: pd.DataFrame
                          ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    fitness_cols = [ f.column_name() for f in self._fitness_fns ]
    population['rank'] = 0
    dominated_solutions = dict()
    num_dominating_solutions = dict()
    fronts = [ pd.DataFrame(columns=population.columns) ]
    for i, curr in population.iterrows():
      dominated_solutions[i] = []
      num_dominating_solutions[i] = 0
      a = curr.loc[fitness_cols].to_numpy()
      for j, other in population.iterrows():
        if i == j:
          continue
        b = other.loc[fitness_cols].to_numpy()
        if Utils.dominates(a, b):
          dominated_solutions[i].append(j)
        if Utils.dominates(b, a):
          num_dominating_solutions[i] += 1
      if num_dominating_solutions[i] == 0:
        population.at[i, 'rank'] = curr['rank'] = 1
        fronts[0] = pd.concat([ fronts[0], curr.to_frame().T ])
    k = 0
    while len(fronts[k]) > 0:
      new_front = pd.DataFrame(columns=population.columns)
      for i, curr in fronts[k].iterrows():
        for j, other in population.loc[dominated_solutions[i]].iterrows():
          num_dominating_solutions[j] -= 1
          if num_dominating_solutions[j] == 0:
            population.at[j, 'rank'] = other['rank'] = k + 2
            new_front = pd.concat([ new_front, other.to_frame().T ])
      k += 1
      fronts.append(new_front)
    return population, fronts[:-1] # the last front is always empty
