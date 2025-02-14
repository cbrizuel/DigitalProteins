from typing import List, Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
import os
import shutil
import glob





def df_append(df: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """
    Appends the given Series as a new row in the given Data Frame.

    Returns
    -------
    pd.DataFrame
        The modified Data Frame.
    """
    return pd.concat([ df, series.to_frame().T ], ignore_index=True)



def pad_zeroes(n: int) -> str:
    """
    Adds up to three zeroes to the left of the integer specified by `n`
    if `n < 1000`.

    If `n >= 1000`, then this function simply casts the number `n` to a string.

    Parameters
    ----------
    n : int
        The number to be padded with zeroes.

    Returns
    -------
    str
        A string containing the padded number `n`.
    """
    result = f'{n}'
    if n < 1000: result = f'{0}{n}'
    if n < 100: result = f'{0}{result}'
    if n < 10: result = f'{0}{result}'
    return result



def coin_toss(rng: np.random.Generator,
              bias: float = 0.5
              ) -> bool:
    """
    Randomly produce a boolean value.

    Parameters
    ----------
    rng : numpy.random.Generator
        The pseudo-random number generator.
    bias : float, optional
        The probability for producing `True`; the default is 0.5. 
        The probability for producing `False` will be computed as `1.0 - bias`.

    Returns
    -------
    bool
        A randomly produced value.
    """
    return rng.choice([ True, False ], p=[ bias, 1. - bias ])



def dominates(a: npt.NDArray[np.float64], 
              b: npt.NDArray[np.float64]
              ) -> bool:
    """
    Returns `True` if vector `a` dominates vector `b`, or `False` 
    otherwise. 
    
    We say vector `a` _dominates_ vector `b` when, for all
    components _x_ in `a` and all components _y_ in `b`, it is the case that
    _x_ >= _y_, and there exists at least one component _z_ in `a` such that 
    _z_ > _y_ for all components _y_.

    Note that if `dominates(a, b)` returns `False`, that does not imply that 
    `dominates(b, a)` will return `True`.
    
    Parameters
    ----------
    a : numpy.typing.NDArray[numpy.float64]
        A vector of real numbers to be compared.
    b : numpy.typing.NDArray[numpy.float64]
        The other vector of real numbers to be compared.

    Returns
    -------
    bool
        Returns `True` if vector `a` dominates over vector `b`, or `False` 
        otherwise.
    """
    return np.all(a >= b) and np.any(a > b)



def is_sorted_before(a: pd.Series,
                     b: pd.Series,
                     columns: List[str],
                     ascending: List[bool]
                     ) -> bool:
    """
    Returns `True` if series `a` must be sorted before series `b`, or `False` 
    otherwise, according to the values of the columns specified by `columns`
    in the specified by `ascending`.

    It is _assumed_ that both `a` and `b` contain the same columns.

    Parameters
    ----------
    a, b : pd.Series
        The series to be compared to determine their relative order.
    columns : List[str]
        The columns which values will be considered to determine the relative
        order of `a` and `b`.
    ascending : List[bool]
        If `True`, it indicates that the values in the corresponding column will 
        be sorted in _ascending_ order. And if `False`, indicates that the values 
        will be sorted in _descending_ order. It is _assumed_ that both `columns`
        and `ascending` have the same length.

    Returns
    -------
    bool
        Returns `True` if series `a` must be sorted before series `b`, or 
        `False` otherwise.
    """
    for k, asc in zip(columns, ascending):
        if a[k] == b[k]: continue
        return a[k] < b[k] if asc else a[k] > b[k]
    return True



def merge(a: pd.DataFrame,
          b: pd.DataFrame,
          columns: List[str],
          ascending: List[bool],
          size: int = 0
          ) -> pd.DataFrame:
    """
    Merges the sorted data frames `a` and `b` into a new sorted data frame.

    It is _assumed_ that both `a` and `b` are sorted and that they both contain
    the same columns. 

    Parameters
    ----------
    a, b : pd.DataFrame
        The data frames to be merged.
    columns : List[str]
        The columns which values will be considered to determine the relative
        order of `a` and `b`.
    ascending : List[bool]
        If `True`, it indicates that the values in the corresponding column will 
        be sorted in _ascending_ order. And if `False`, indicates that the values 
        will be sorted in _descending_ order. It is _assumed_ that both `columns`
        and `ascending` have the same length.
    size : int, optional
        If a value greater than zero is provided, then only the first `size` rows
        in both `a` and `b` will be merged. If the value zero is provided, then
        all the rows in both `a` and `b` will be merged. Default value is zero.

    Returns
    -------
    pd.DataFrame
        A new sorted data frame that contains the same rows as `a` and `b`.
    """
    if not size: size = len(a) + len(b)
    merged = pd.DataFrame(columns=a.columns)
    i, j, n, m = 0, 0, len(a), len(b)
    while i < n and j < m and len(merged) < size:
        row_a, row_b = a.iloc[i], b.iloc[j]
        if is_sorted_before(row_a, row_b, columns, ascending):
            merged = pd.concat([ merged, a.iloc[i].to_frame().T ],
                               ignore_index=True)
            i += 1
        else:
            merged = pd.concat([ merged, b.iloc[j].to_frame().T ],
                               ignore_index=True)
            j += 1
    unmerged = size - len(merged)
    if i < n and unmerged > 0:
        merged = pd.concat([ merged, a.iloc[i:i + unmerged] ], 
                           axis=0, 
                           ignore_index=True)
    if j < m and unmerged > 0:
        merged = pd.concat([ merged, b.iloc[j:j + unmerged] ], 
                           axis=0, 
                           ignore_index=True)
    return merged



def find_top_solution(population: pd.DataFrame,
                      columns: List[str],
                      ascending: List[bool]
                      ) -> pd.Series:
    """
    Returns the data of the fittest individual in the given population.

    Parameters
    ----------
    population : pd.DataFrame
        The population to be searched.
    columns : List[str]
        The columns which values will be considered to determine the relative
        order of `a` and `b`.
    ascending : List[bool]
        If `True`, it indicates that the values in the corresponding column will 
        be sorted in _ascending_ order. And if `False`, indicates that the values 
        will be sorted in _descending_ order. It is _assumed_ that both `columns`
        and `ascending` have the same length.

    Returns
    -------
    pd.Series
        A copy of the data of the found individual. Modifications on this data will 
        not be reflected in the original individual's data.
    """
    top_solution = population.iloc[0].copy()
    for _, row in population.iterrows():
        if is_sorted_before(top_solution, row, columns, ascending):
            continue
        top_solution = row.copy()
    return top_solution



def clone_workspace(source_dir: str, 
                    destination_dir: str
                    ) -> None:
    pdbs_dir = f'{destination_dir}/pdbs'
    os.makedirs(pdbs_dir)
    source_pdbs_dir = f'{source_dir}/pdbs'
    if os.path.isdir(source_pdbs_dir) and len(os.listdir(source_pdbs_dir)) > 0:
        for pdb_path in glob.glob(f'{source_dir}/pdbs/prot_0001_*.pdb'):
            shutil.copy(pdb_path, pdbs_dir)
    populations_dir = f'{destination_dir}/populations'
    os.makedirs(populations_dir)
    shutil.copy(f'{source_dir}/populations/pop_0001.csv', 
                populations_dir)
    # remove fitness columns
    df = pd.read_csv(f'{populations_dir}/pop_0001.csv')
    df = df[[ 'generation_id', 'sequence_id', 'sequence', 'survivor' ]]
    df.to_csv(f'{populations_dir}/pop_0001.csv', index=False)
    shutil.copy(f'{source_dir}/initial_rng_state.json', destination_dir)
    shutil.copy(f'{source_dir}/settings.json', destination_dir)
    for pdb_path in glob.glob(f'{source_dir}/*.pdb'):
        shutil.copy(pdb_path, destination_dir)
    for fasta_path in glob.glob(f'{source_dir}/*.fasta'):
        shutil.copy(fasta_path, destination_dir)
    restrictions_path = f'{source_dir}/sequence_restrictions.json'
    if os.path.isfile(restrictions_path):
        shutil.copy(restrictions_path, destination_dir)



def workspace_average_fitness(workspace_dir: str,
                              fitness_column: Optional[str] = None
                              ) -> pd.DataFrame:
    stats = pd.DataFrame(columns=[ "generation_id", "average_fitness"])
    pop_files = os.listdir(f"{workspace_dir}/populations")
    for i in range(len(pop_files)):
        pop_path = f"{workspace_dir}/populations/pop_{pad_zeroes(i + 1)}.csv"
        df = pd.read_csv(pop_path)
        df = df[df["survivor"] == True]
        if fitness_column is None:
            mask = np.array(list(map(lambda c: c.find("Fitness.") != -1, df.columns)))
            indices = np.array(list(range(1, len(df.columns) + 1)))
            j = (mask * indices).sum()
            if j == 0: raise RuntimeError
            fitness_column = list(df.columns)[j - 1]
        avg_fitness = df[fitness_column].mean()
        row = pd.Series({ "generation_id": i + 1, "average_fitness": avg_fitness })
        stats = pd.concat([ stats, row.to_frame().T ], ignore_index=True)
    return stats
