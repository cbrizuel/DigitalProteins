from ..Metric import Metric
from typing import List, Optional
import statistics





class Lddt(Metric):

  def _params(self) -> dict:
    params = super()._params()
    params['cutoffs'] = self._cutoffs
    params['radius'] = self._radius
    return params
  


  def __init__(self,
               cutoffs: List[float] = [ 0.5, 1.0, 2.0, 4.0 ],
               radius: float = 15,
               column: Optional[str] = None
               ) -> None:
    super().__init__(column)
    self._radius = radius
    self._cutoffs = cutoffs



  def __call__(self, **kwargs) -> float:
    model = kwargs['model']
    reference = kwargs['reference']
    
    def is_preserved(i: int, j: int, cutoff: float) -> bool:
      a = model[i]
      b = model[j]
      x = reference[i]
      y = reference[j]
      return abs((x - y) - (a - b)) <= cutoff
    
    reference_set = {
      frozenset([ i, j ])
      for i, a in enumerate(reference)
      for j, b in enumerate(reference)
      if a - b > 0 and i // 3 != j // 3 and a - b <= self._radius
    }
    model_set = {
        frozenset([ i, j ])
        for i, a in enumerate(model)
        for j, b in enumerate(model)
        if a - b <= self._radius and frozenset([ i, j ]) in reference_set
    }
    
    n = len(reference_set)
    return statistics.mean([
      sum([ is_preserved(list(x)[0], list(x)[1], c) for x in model_set ]) / n
      for c in self._cutoffs
    ])
