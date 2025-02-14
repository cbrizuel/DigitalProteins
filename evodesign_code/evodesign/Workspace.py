import evodesign.Population as Population
import pandas as pd
import json
import os
import shutil
from typing import Optional, Dict, List





class Workspace:

  def __init__(self, 
               root_dir: str,
               target_pdb_path: str,
               target_fasta_path: Optional[str] = None
               ) -> None:
    """
    Interface for managing the file system folder where all the output files 
    will be stored.

    Parameters
    ----------
    path : str
        The folder where all the output files and folders will be stored.
    target_pdb_path : str
        The path to the PDB file containing the target protein backbone.
    target_fasta_path : str, optional
        The path to the FASTA file containing the amino acid sequence of the 
        target protein. Default is `None`.
    """
    self.root_dir = root_dir
    self.target_pdb_path = target_pdb_path
    self.populations_dir = f'{self.root_dir}/populations'
    self.pdbs_dir = f'{self.root_dir}/pdbs'
    self.ilearn_dir = f'{self.root_dir}/ilearn'
    self.esm2_dir = f'{self.root_dir}/esm2'
    self.target_fasta_path = target_fasta_path
  


  def save_commit_hash(self) -> None:
    """
    Saves a text file in the workspace folder indicating the version of this
    program that was used to produce the results contained in it.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    filepath = f'{self.root_dir}/commit_hash.txt'
    with open(filepath, 'wt', encoding='utf-8') as txt_file:
      txt_file.write('https://github.com/coliva92/evodesign/commit/aa9e148afe89289c4ee5d53c36bb4f386cbf7693\n')



  def save_target_pdb(self) -> None:
    """
    Makes a copy in the workspace folder of the target PDB file.
    """
    name = os.path.basename(self.target_pdb_path)
    destination = f'{self.root_dir}/{name}'
    os.makedirs(self.root_dir, exist_ok=True)
    try:
      shutil.copy(self.target_pdb_path, destination)
    except shutil.SameFileError:
      pass
    return

  

  def save_target_fasta(self) -> None:
    """
    Makes a copy in the workspace folder of the target FASTA file. 
    If no such file was provided, then this function does nothing.
    """
    if self.target_fasta_path is None: return
    name = os.path.basename(self.target_fasta_path)
    destination = f'{self.root_dir}/{name}'
    os.makedirs(self.root_dir, exist_ok=True)
    try:
      shutil.copy(self.target_fasta_path, destination)
    except shutil.SameFileError:
      pass
    return
  


  def load_target_fasta(self):
    if self.target_fasta_path is None: return None
    for line in open(self.target_fasta_path, 'rt', encoding='utf-8'):
      if line.find('>') > -1: continue
      sequence = line
      break
    return sequence



  def save_population(self, 
                      population: pd.DataFrame,
                      temporary: bool = False
                      ) -> None:
    """
    Saves the given population in a CSV file in the workspace.

    Parameters
    ----------
    population : pandas.DataFrame
        The population to be saved.
    temporary : bool, optional
        Indicates if the population file is meant to stored temporarily or 
        permanently. Populations meant to be stored permanently will be stored
        in their dedicated folder in the workspace, while populations meant to
        be stored temporarily will be stored in the root directory of the 
        workspace. Only a single population file may be stored in the root
        directory, thus, any subsequent calls to this function using
        `temporary = True` will overwrite the population file previously
        stored in the root directory. The default is False.
    """
    os.makedirs(self.populations_dir, exist_ok=True)
    filename = f'{self.root_dir}/.next_population' \
               if temporary \
               else f'{self.populations_dir}/{Population.file_path(population)}'
    population.to_csv(filename, index=False)
    
  

  def load_population(self, temporary: bool = False) -> pd.DataFrame:
    """
    Loads the population data from a CSV file in the workspace. If no population
    file is found in the current workspace, then an empty DataFrame is returned.

    Returns
    -------
    temporary : bool, optional
        Indicates if the population file should be loaded from the  
        temporary storage or from the permanent storage. The default is False.
        When loading from the permanent storage, only the population file of the 
        latest generation will be loaded.
    pandas.DataFrame
        The data for the population found.
    """
    if temporary:
      return self._load_from_root_dir()
    return self._load_from_populations_dir()
  


  def delete_temporary_population(self) -> None:
    """
    Deletes the temporary population CSV file from the workspace.
    If such file does not exist, then nothing is done.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    filename = f'{self.root_dir}/.next_population'
    if os.path.isfile(filename):
      os.remove(filename)
  


  def save_rng_state(self, 
                     state: dict, 
                     checkpoint: bool = True
                     ) -> None:
    """
    Saves the given RNG state in a JSON file in the workspace.

    Parameters
    ----------
    state : dict
        The RNG state to be saved.
    checkpoint : bool, optional
        If `True`, then the given state will be saved as a "checkpoint"
        file, so that the evolutionary algorithm can use the RNG from the 
        last known position in the random number sequence. Otherwise, the
        state will be saved as the state that the RNG had before running
        the evolutionary algorithm from the first generation. 
        The default is `True`.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    filename = f'{self.root_dir}/.rng_state_checkpoint' \
               if checkpoint \
               else f'{self.root_dir}/initial_rng_state.json'
    # TODO validate that the state has the correct format
    with open(filename, 'wt', encoding='utf-8') as json_file:
      json.dump(state, json_file)
  


  def load_rng_state(self, checkpoint: bool = True) -> Optional[dict]:
    """
    Loads the RNG state from a previously saved JSON file in the workspace. 
    If no such file is found, then `None` is returned.

    Parameters
    ----------
    checkpoint : bool, optional
        If `True`, the workspace is first searched for the JSON file containing 
        the last known RNG state. If `False`, no such search is performed and 
        instead the workspace is immediately searched for the JSON file 
        containing the initial RNG state at the beginning of the evolutionary 
        algorithm. The default is `True`.

    Returns
    -------
    dict | None
        The state object found.
    """
    filename = f'{self.root_dir}/.rng_state_checkpoint'
    if not (checkpoint and os.path.isfile(filename)):
      filename = f'{self.root_dir}/initial_rng_state.json'
      if not os.path.isfile(filename):
        return None
    with open(filename, 'rt', encoding='utf-8') as json_file:
      state = json.load(json_file)
    # TODO validar que el json cargado tenga el formato correcto
    return state
  


  def save_statistics(self, stats: pd.DataFrame) -> None:
    """
    Saves the given population statistics data into a CSV file in the workspace.

    Parameters
    ----------
    stats : pandas.DataFrame
        The statistics data that will be saved.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    filename = f'{self.root_dir}/statistics.csv'
    stats.to_csv(filename, index=False)
  


  def load_statistics(self) -> pd.DataFrame:
    """
    Loads the population statistics data from a CSV file in the workspace.
    If no such file is found, then an empty `DataFrame` is returned.

    Returns
    -------
    pandas.DataFrame
        The statistics data found.
    """
    filename = f'{self.root_dir}/statistics.csv'
    if not os.path.isfile(filename):
      return pd.DataFrame()
    return pd.read_csv(filename)



  def save_settings(self, settings: dict) -> None:
    """
    Saves the given algorithm settings in a JSON file in the workspace.

    Parameters
    ----------
    settings : dict
        The algorithm settings to be saved.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    filename = f'{self.root_dir}/settings.json'
    with open(filename, 'wt', encoding='utf-8') as json_file:
      json_file.write(json.dumps(settings, indent=4) + '\n')
    
  

  def save_sequence_restrictions(self, restrictions_desc: Dict[int, List[str]]) -> None:
    """
    Saves the given amino acid sequence letter restrictions in a JSON file in the
    workspace.

    Parameters
    ----------
    restrictions_desc : Dict[int, List[str]]
        The restrictions description to be saved.
    """
    os.makedirs(self.root_dir, exist_ok=True)
    file_path = f'{self.root_dir}/sequence_restrictions.json'
    with open(file_path, 'wt', encoding='utf-8') as json_file:
      json_file.write(json.dumps(restrictions_desc, indent=4) + '\n')
  


  def delete_pdb_files(self) -> None:
    if not os.path.isdir(self.pdbs_dir):
      return
    shutil.rmtree(self.pdbs_dir)
  


  def _load_from_populations_dir(self) -> pd.DataFrame:
    if not os.path.isdir(self.populations_dir):
      return pd.DataFrame()
    filenames = os.listdir(self.populations_dir)
    if not filenames or len(filenames) == 0:
      return pd.DataFrame()
    filename = f'{self.populations_dir}/{sorted(filenames)[-1]}'
    return pd.read_csv(filename)
  


  def _load_from_root_dir(self) -> pd.DataFrame:
    filename = f'{self.root_dir}/.next_population'
    if not os.path.isfile(filename):
      return pd.DataFrame()
    return pd.read_csv(filename)
