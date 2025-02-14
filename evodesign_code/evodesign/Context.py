from typing import Optional
import evodesign.Chain as Chain
import numpy as np
import math
from typing import Any, Optional
from .Workspace import Workspace
import evodesign.Sequence as Sequence
import json





class Context:

    @classmethod
    def create(cls, 
               target_pdb_path: str,
               target_fasta_path: Optional[str] = None,
               num_generations: int = 0,
               seq_allowed_letters_json_path: Optional[str] = None
               ) -> Any: # Context
        """
        Target protein data and other interfaces that might be required by 
        the evolutionary algorithm at any point of its execution. A Context can
        be shared among multiple different algorithms.

        Parameters
        ----------
        target_pdb_path : str
            The path to the PDB file containing the target protein backbone.
        target_fasta_path : str, optional
            The path to the FASTA file containing the amino acid sequence of the 
            target protein. Default is `None`.
        num_generations : int, optional
            The amount of generations for which the algorithm should be 
            executed (unless another termination condition is met first).
            If zero, then the algorithm will run for the configured max number
            of generations. Default is 0.
        seq_allowed_letters_json_path : str, optional
            The path to the JSON file describing which amino acid letters are allowed
            for certain positions in the designed sequences. If none is provided, then
            no letter restrictions will be imposed for any position. Default is `None`.
        """
        context = Context()
        context._target_pdb_path = target_pdb_path
        context._target_fasta_path = target_fasta_path
        
        # load the target protein
        context.ref_structure = Chain.load_structure(target_pdb_path)
        context.ref_backbone = Chain.backbone_atoms(context.ref_structure)

        # load the target sequence if one was provided
        context.ref_sequence = None
        if target_fasta_path is not None:
            for line in open(target_fasta_path, 'rt', encoding='utf-8'):
                if line.find('>') != -1: continue
                context.ref_sequence = line.strip()
                break
        
        # compute the sequence length
        context.sequence_length = Chain.length(context.ref_structure)
        
        # set the limit for how many generations to execute
        context.num_generations = math.inf
        if num_generations is not None and num_generations > 0:
            context.num_generations = num_generations
        
        # load the sequence restrictions
        if seq_allowed_letters_json_path is None:
            desc = Sequence.allowed_letters_desc(context.sequence_length)
        else:
            with open(seq_allowed_letters_json_path, 'rt', encoding='utf-8') as json_file:
                # TODO revisar que el formato del JSON esté correcto
                desc = json.load(json_file)
            desc = {
                int(float(key)): value
                for key, value in desc.items()
            }
            desc = Sequence.allowed_letters_desc(context.sequence_length, desc)
        context.sequence_allowed_letters = desc
        return context
    


    def __init__(self) -> None:
        """
        Creates an empty context.
        """
        self._target_pdb_path = None
        self._target_fasta_path = None
        self.ref_structure = None
        self.ref_backbone = None
        self.ref_sequence = None
        self.sequence_length = None
        self.num_generations = None
        self.rng = None
        self.workspace = None
        self.sort_columns = None
        self.sort_ascending = None
        self.sequence_allowed_letters = None
    


    def duplicate(self) -> Any: # Context
        """
        Creates a new context instance that shares the same data as the calling 
        instance.

        Returns
        -------
        Context
            The duplicated context instance.
        """
        context_copy = Context()
        context_copy._target_pdb_path = self._target_pdb_path
        context_copy._target_fasta_path = self._target_fasta_path
        context_copy.ref_structure = self.ref_structure
        context_copy.ref_backbone = self.ref_backbone
        context_copy.ref_sequence = self.ref_sequence
        context_copy.sequence_length = self.sequence_length
        context_copy.num_generations = self.num_generations
        context_copy.rng = self.rng
        context_copy.workspace = self.workspace
        context_copy.sort_columns = self.sort_columns
        context_copy.sort_ascending = self.sort_ascending
        context_copy.sequence_allowed_letters = self.sequence_allowed_letters
        return context_copy
    


    def init_workspace(self, workspace_root: str) -> None:
        """
        Initializes the workspace interface held by this context instance.

        Parameters
        ----------
        workspace_root : str
            Path to the folder where all the output files generated by the 
            evolutionary algorithm will be stored.
        """
        self.workspace = Workspace(workspace_root, 
                                   self._target_pdb_path, 
                                   self._target_fasta_path)
    


    def init_rng(self, seed: Optional[int] = None) -> None:
        """
        Initializes the RNG held by this context instance.

        If the workspace was already initialized, this function attempts to load
        the RNG state from a checkpoint or initial file already stored in said 
        workspace. If the workspace haven't been initialized yet, or the 
        workspace does not yet exists in the file system, then a new RNG 
        instance is created with the default seed and its state is stored in 
        the workspace. 

        Parameters
        ----------
        seed : int
            A custom seed for the RNG.
        """
        self.rng = np.random.default_rng(seed)
        if self.workspace is None: return
        state = self.workspace.load_rng_state(checkpoint=True)
        if state is None:
            state = self.workspace.load_rng_state(checkpoint=False)
        if state is None:
            self.workspace.save_rng_state(self.rng.bit_generator.state, 
                                          checkpoint=False)
        else:
            self.rng.bit_generator.state = state
