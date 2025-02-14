from ..Rastrigin import Rastrigin





class Rastrigin2(Rastrigin):

    def _normalize(self, n: int, n_ref: int, n_min: int, n_max: int) -> float:
        return (n - n_min) / (n_max - n_min)
