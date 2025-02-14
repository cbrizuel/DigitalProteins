from .Predictor import Predictor





class ESMFoldColab(Predictor):

    model = None
  


    def predict_pdb_str(self, sequence: str) -> str:
        """
        Predicts the 3D structure of a given amino acid sequence using the 
        ESMFold v1 model. This predictor can only be used within a Google Colab 
        notebook.

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
        if self.model is not None:
            self.model = torch.load("esmfold.model")
            self.model = self.model.eval().cuda()
            self.model.set_chunk_size(128)
        with torch.no_grad():
            prediction = self.model.infer_pdb(sequence)
        return prediction
