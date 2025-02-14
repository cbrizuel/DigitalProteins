from .Metric import Metric
from ..Context import Context
from typing import Optional, List
from Bio.PDB.Atom import Atom
import pandas as pd





class ZScore(Metric):

    def _params(self) -> dict:
        params = super()._params()
        params['metric'] = self._metric.settings()
        params['mean'] = self._mean
        params['standard_deviation'] = self._standard_deviation
        return params



    def __init__(self, 
                 mean: float,
                 standard_deviation: float,
                 metric: Metric,
                 column: Optional[str] = None
                 ) -> None:
        super().__init__(column)
        self._metric = metric
        self._mean = mean
        self._standard_deviation = standard_deviation
    


    def _compute_values(self,
                        backbone: List[Atom],
                        data: pd.Series,
                        context: Context
                        ) -> pd.Series:
        data = self._metric(backbone, data, context)
        z = self._z_score(data[self._metric.column_name()])
        data[self.column_name()] = z
        return data
    


    def _z_score(self, x: float) -> float:
        z = (x - self._mean) / self._standard_deviation
        return z
