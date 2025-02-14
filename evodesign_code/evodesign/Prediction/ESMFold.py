from .Predictor import Predictor
from typing import Optional





class ESMFold(Predictor):

    model = None



    def _params(self) -> dict:
        return {
            'gpu_device': self._gpu_device
        }
  


    def __init__(self, 
                 gpu_device: Optional[str] = "cuda:0"
                 ) -> None:
        """
        Interface for interacting with the ESMFold v1 model for protein structure 
        prediction.

        Parameters
        ----------
        gpu_device : str, optional
            If not `None`, then the predictor model will run in the 
            specified GPU device (if Nvidia CUDA is available). If this
            value is `None` or Nvidia CUDA is not available, the model will 
            run entirely on the CPU. Default is "cuda:0".
        """
        super().__init__()
        self._gpu_device = gpu_device



    def predict_pdb_str(self, sequence: str) -> str:
        """
        Predicts the 3D structure of a given amino acid sequence using the 
        ESMFold v1 model.

        Parameters
        ----------
        sequence : str
            The amino acid sequence which structure will be predicted. Each residue
            must be represented with a single letter corresponding to one of the
            20 essential amino acids.
        
        Returns
        -------
        str
            The predicted 3D structure codified in a string as a PDB file.
        """
        import torch
        if self.model is None:
            import esm
            self.model = esm.pretrained.esmfold_v1()
            self.model.eval()
            if torch.cuda.is_available() and self._gpu_device is not None:
                device = torch.device(self._gpu_device)
                self.model = self.model.to(device)
                self.model.set_chunk_size(128)
        with torch.no_grad():
            prediction = self.model.infer_pdb(sequence)
        return prediction
