from .Predictor import Predictor
from ..Workspace import Workspace
import subprocess
import os





class AlphaFold(Predictor):
  
    def _params(self) -> dict:
        return {
            'fakemsa_script_path': self._fakemsa_script_path,
            'run_af_script_path': self._run_af_script_path,
            'mgnify_database_path': self._mgnify_database_path,
            'data_dir': self._data_dir,
            'max_template_date': self._max_template_date,
            'model_preset': self._model_preset,
            'db_preset': self._db_preset
        }
  


    # mgnify_database_path = /media/biocomp/My Passport/mgnify/mgy_clusters_2018_12.fa
    # data_dir = /media/biocomp/My Passport/reduced_dbs
    def __init__(self,
                 fakemsa_script_path: str,
                 run_af_script_path: str,
                 mgnify_database_path: str,
                 data_dir: str,
                 max_template_date: str = '2020-05-14',
                 model_preset: str = 'monomer',
                 db_preset: str = 'reduced_dbs'
                 ) -> None:
        """
        Interface for interacting with the AlphaFold2 model for protein structure 
        prediction.

        Parameters
        ----------
        fakemsa_script_path : str
            The path to the script provided from 
            https://github.com/Zuricho/ParaFold_dev/blob/main/parafold/create_fakemsa.py.
            When doing "de novo" protein design with AlphaFold, we usually want
            to skip the MSA procedure. In order to do this, we provide AlphaFold
            with an empty MSA file. The aformentioned script allow us to create
            such MSA file.
        run_af_script_path : str
            The path to the script that runs AlphaFold.
        mgnify_database_path : str
            The path to the Mgnify data base file. This is required by AlphaFold.
        data_dir : str
            The path to the folder where the genetic and structure data bases 
            used by AlphaFold are stored.
        max_template_date : str
            A date string with format 'YYYY-MM-DD' that designates which templates
            can be used by AlphaFold; only templates that were available in the 
            PDB at this date or earlier can be used. The default is '2020-05-14',
            corresponding to the original CASP14 configuration.
        model_preset : { 'monomer', 'monomer_casp14', 'monomer_ptm', 'multimer' }
            The specific AlphaFold model to use. For single-chained peptides,
            the 'monomer' model is recommended. For multi-chained molecules and
            complexes, use the 'multimer' model instead. The default is 'monomer'.
        db_preset : { 'reduced_dbs', 'full_dbs' }
            Controls the speed and quality of the MSA performed by AlphaFold.
            With the 'reduced_dbs' option, a reduced version of the BFD databases
            will be used. Otherwise, with the 'full_dbs' option, all the genetic
            databases will be used. The default is 'reduced_dbs', since we're
            skipping MSA anyway.
        """
        super().__init__()
        self._fakemsa_script_path = fakemsa_script_path
        self._run_af_script_path = run_af_script_path
        workspace = Workspace.instance()
        self._output_dir = f'{workspace.root_dir}/alphafold_outputs'
        self._mgnify_database_path = mgnify_database_path
        self._data_dir = data_dir
        self._max_template_date = max_template_date
        self._model_preset = model_preset
        self._db_preset = db_preset



    def predict_pdb_str(self, sequence: str) -> str:
        """
        Predicts the 3D structure of a given amino acid sequence using the 
        AlphaFold2 model. The resulting prediction is returned as a string
        containing a PDB file description.

        Parameters
        ----------
        sequence : str
            The amino acid sequence which structure will be predicted. Each residue
            must be represented with a single letter corresponding to one of the
            20 essential amino acids.
        """
        pdb_path = '.temp.pdb'
        self.predict_pdb_file(sequence, pdb_path)
        with open(pdb_path, 'rt', encoding='utf-8') as pdb_file:
            prediction = pdb_file.read()
        os.remove(pdb_path)
        return prediction

    

    def predict_pdb_file(self, 
                         sequence: str, 
                         pdb_path: str
                         ) -> None:
        """
        Predicts the 3D structure of a given amino acid sequence using the 
        AlphaFold2 model. The resulting prediction is stored in a PDB file at
        the specified file path.

        Parameters
        ----------
        sequence : str
            The amino acid sequence which structure will be predicted. Each residue
            must be represented with a single letter corresponding to one of the
            20 essential amino acids.
        pdb_path : str
            The path and name of the PDB file where the predicted structure will
            be stored.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        protein_id = os.path.splitext(os.path.basename(pdb_path))[0]
        workspace = Workspace.instance()
        fasta_path = f'{workspace.root_dir}/{protein_id}.fasta'
        with open(fasta_path, 'wt', encoding='utf-8') as fasta_file:
            fasta_file.write(f'>{protein_id}\n{sequence}\n')
        # run the script for creating an empty MSA
        subprocess.call([
            'python3',
            self._fakemsa_script_path,
            f'--fasta_paths={fasta_path}',
            f'--output_dir={self._output_dir}'
        ])
        for line in self._run_alphafold_docker(fasta_path):
            print(line, end='')
        prediction_pdb = f'{self._output_dir}/{protein_id}/ranked_0.pdb'
        os.symlink(prediction_pdb, pdb_path)
        os.remove(fasta_path)



    def _run_alphafold_docker(self, fasta_path: str):
        cmd = [
            'python3',
            self._run_af_script_path,
            '--use_precomputed_msas=True',
            f'--fasta_paths={fasta_path}',
            f'--max_template_date={self._max_template_date}',
            f'--model_preset={self._model_preset}',
            f'--db_preset={self._db_preset}',
            f'--output_dir={self._output_dir}',
            f'--mgnify_database_path={self._mgnify_database_path}',
            f'--data_dir={self._data_dir}'
        ]
        popen = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)
    