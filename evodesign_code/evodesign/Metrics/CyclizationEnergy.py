from .Metric import Metric
from .RosettaEnergyFunction import RosettaEnergyFunction
from typing import Optional, List
from Bio.PDB.Atom import Atom
import pandas as pd
from ..Context import Context
import numpy as np





class CyclizationEnergy(Metric):

    MEAN = 849.47
    STANDARD_DEVIATION = 118.4751 / 4.0
    VARIANCE = STANDARD_DEVIATION ** 2



    def _params(self) -> dict:
        params = super()._params()
        params["metric"] = self._metric.settings()
        return params



    def __init__(self,
                 metric: Optional[RosettaEnergyFunction] = None,
                 column: Optional[str] = None
                 ) -> None:
        super().__init__(column)
        if metric is None:
            metric = RosettaEnergyFunction()
        self._metric = metric


    def _compute_values(self, 
                        backbone: List[Atom],
                        data: pd.Series,
                        context: Context
                        ) -> pd.Series:
        data = self._metric(backbone, data, context)
        energy = data[self._metric.column_name()]
        data[self.column_name()] = self._normalize(energy)
        return data
    


    def _normalize(self, x: float) -> float:
        return np.exp(-(((x - self.MEAN) ** 2) / self.VARIANCE))
