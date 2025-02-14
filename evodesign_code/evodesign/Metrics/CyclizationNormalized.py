from .Metric import Metric
from .Cyclization import Cyclization as Cyclization
from .ZScore import ZScore
from ..Context import Context
from typing import Optional, List
from Bio.PDB.Atom import Atom
import pandas as pd
import numpy as np





class CyclizationNormalized(Metric):

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
            metric = ZScore(mean=1.3248119, 
                            standard_deviation=0.10498072, 
                            metric=Cyclization())
        self._metric = metric
        self._scaling_factor = scaling_factor
    


    def _compute_values(self, 
                        backbone: List[Atom],
                        data: pd.Series,
                        context: Context
                        ) -> pd.Series:
        data = self._metric(backbone, data, context)
        cyc = self._normalize(data[self._metric.column_name()])
        data[self.column_name()] = cyc
        return data



    def _normalize(self, z: float) -> float:
        return 1.0 / (1.0 + self._scaling_factor * np.abs(z))
