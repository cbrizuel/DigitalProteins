from .Metric import Metric
from .RosettaEnergyFunction import RosettaEnergyFunction
from .ZScore import ZScore
from ..Context import Context
from typing import Optional, List
from Bio.PDB.Atom import Atom
import pandas as pd
import numpy as np





class RosettaEnergyNormalized(Metric):

    def _params(self) -> dict:
        params = super()._params()
        params["scaling_factor"] = self._scaling_factor
        params["metric"] = self._metric.settings()
        return params



    def __init__(self, 
                 scaling_factor: float = 1.0,
                 metric: Optional[ZScore] = None,
                 column: Optional[str] = None
                 ) -> None:
        super().__init__(column)
        if metric is None:
            metric = ZScore(mean=849.47, 
                            standard_deviation=118.8751, 
                            metric=RosettaEnergyFunction())
        self._metric = metric
        self._scaling_factor = scaling_factor
    


    def _compute_values(self, 
                        backbone: List[Atom],
                        data: pd.Series,
                        context: Context
                        ) -> pd.Series:
        data = self._metric(backbone, data, context)
        energy = self._normalize(data[self._metric.column_name()])
        data[self.column_name()] = energy
        return data



    def _normalize(self, z: float) -> float:
        return 1.0 / (1.0 + self._scaling_factor * np.abs(z))
