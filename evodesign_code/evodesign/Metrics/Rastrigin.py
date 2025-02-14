from .Metric import Metric
from typing import List, Optional
import evodesign.Sequence as Sequence
import math
import pandas as pd
from Bio.PDB.Atom import Atom
from ..Context import Context
import numpy as np
import numpy.typing as npt





class Rastrigin(Metric):

  """[Función de Rastrigin](https://www.sfu.ca/~ssurjano/rastr.html) para 
  realizar pruebas sin utilizar directamente el predictor de la estructura de 
  una proteína al calcular la aptitud. Esta función debe utilizarse en conjunto 
  con `Predictor_Null`
  """
  


  def _params(self) -> dict:
    params = super()._params()
    params['sub_sequence_length'] = self._SUB_SEQ_LENGTH
    return params
  


  def __init__(self,
               column: Optional[str] = None,
               sub_sequence_length: int = 5
               ) -> None:
    super().__init__(column)
    if sub_sequence_length < 0:
      sub_sequence_length = -sub_sequence_length
    if sub_sequence_length % 2 == 0:
      sub_sequence_length += 1
    self._SUB_SEQ_LENGTH = sub_sequence_length
    self._MIN_SUB_SEQ_VALUE = self._to_ordinal_value("".join(self._SUB_SEQ_LENGTH * [ "A" ]))
    self._MAX_SUB_SEQ_VALUE = self._to_ordinal_value("".join(self._SUB_SEQ_LENGTH * [ "Y" ]))
    self._LOWER_BOUND = -5.12
    self._UPPER_BOUND = 5.12
    self._ref_ordinal_values = None



  def _compute_values(self, 
                      backbone: List[Atom],
                      data: pd.Series,
                      context: Context
                      ) -> pd.Series:
    if self._ref_ordinal_values is None:
      self._ref_ordinal_values = self._ordinal_values(context.ref_sequence)
    ordinal_values = self._ordinal_values(data["sequence"])
    x = self._to_rastrigin_domain(ordinal_values, self._ref_ordinal_values)
    data[self.column_name()] = self._compute_rastrigin_value(x)
    return data



  def compute_rastrigin_from_sequences(self, 
                                       sequence: str, 
                                       ref_sequence: str
                                       ) -> float:
    # La función de Rastrigin toma como entrada un vector x de
    # d componentes, donde cada componente x_i es un número real en el 
    # intervalo cerrado [-5.12, 5.12]. El óptimo global se encuentra en 
    # x = (0, ..., 0), y su valor es f(x) = 0.
    # El objetivo de esta función es transformar una secuencia de 
    # aminoácidos a un vector de números reales para la función de Rastrigin, 
    # de tal manera que el vector para el óptimo global corresponda con la 
    # secuencia de la estructura objetivo.
    ref_ordinal_values = self._ordinal_values(ref_sequence)
    ordinal_values = self._ordinal_values(sequence)
    x = self._to_rastrigin_domain(ordinal_values, ref_ordinal_values)
    return self._compute_rastrigin_value(x)



  def _ordinal_values(self, sequence: str) -> List[int]:
    # Dado que el conjunto de aminoácidos que se permiten en cada residuo es
    # discreto y finito, si deseamos asociar cada aminoácido en dicho conjunto 
    # con un número real en el intervalo [-5.12, 5.12], inevitablemente algunos 
    # de los números en este intervalo no podrán ser representados (por la 
    # discretización). Este problema se amortigua al usar una subsecuencia o 
    # ventana (en lugar de un solo aminoácido) por cada posición o residuo en 
    # la secuencia.
    return list(map(self._to_ordinal_value, self._get_sub_sequences(sequence)))
  


  def _get_sub_sequences(self, sequence: str) -> List[str]:
    # en este caso, las subsecuencias no se traslapan
    sub_sequences = [
      sequence[i : i + self._SUB_SEQ_LENGTH]
      for i in range(0, len(sequence), self._SUB_SEQ_LENGTH)
    ]
    # si el tamaño de las subsecuencias no alcanza a cubrir por completo la secuencia
    # de entrada, completamos la última subsecuencia tomando los primeros caracteres
    # de la secuencia de entrada
    n = len(sequence) % self._SUB_SEQ_LENGTH
    if n > 0:
      sub_sequences[-1] += sequence[0 : self._SUB_SEQ_LENGTH - n]
    return sub_sequences
  


  def _to_ordinal_value(self, sub_sequence: str) -> int:
    # Esta función convierte la subsecuencia de entrada en un valor ordinal de 
    # base 10 usando el valor en ASCII de cada letra
    ordinal_value = sum([
      ord(letter) * len(Sequence.AMINO_ACIDS) ** i
      for i, letter in enumerate(reversed(sub_sequence))
    ])
    return ordinal_value
  


  def _to_rastrigin_domain(self, 
                           ordinal_values: List[int],
                           ref_ordinal_values: List[int]
                           ) -> npt.NDArray[np.float64]:
    # Esta función toma los valores ordinales de cada subsecuencia (estos valores
    # fueron calculados previamente) y los normaliza llevándolos al intervalo de 
    # [-5.12, 5.12].
    a = self._LOWER_BOUND
    b_minus_a = self._UPPER_BOUND - self._LOWER_BOUND
    n_min = self._MIN_SUB_SEQ_VALUE
    n_max = self._MAX_SUB_SEQ_VALUE
    x = np.array([
      a + b_minus_a * self._normalize(n, n_ref, n_min, n_max)
      for n, n_ref in zip(ordinal_values, ref_ordinal_values)
    ])
    return x
  


  def _normalize(self, n: int, n_ref: int, n_min: int, n_max: int) -> float:
    # Para que esta función pueda usarse en conjunto con la normalización al intervalo
    # [-5.12, 5.12], debe cumplir con las sigs. propiedades:
    # i)   f(n) = 0 si n_min <= n <= n_ref
    # ii)  f(n) = 0.5 si n = n_ref
    # iii) f(n) = 1 si n_ref < n <= n_max
    # donde n_min es el valor ordinal correspondiente a la subsecuencia de puras As (la
    # letra con el valor más pequeño), n_max es el valor ordinal correspondiente a la
    # subsecuencia de puras Ys (la letra con el valor más grande), y n_ref es el valor
    # ordinal correspondiente a la subsecuencia en la secuencia de referencia.
    # Obsérvese que se está suponiendo que n_min <= n_mid <= n_max y que 
    # n_min <= n <= n_max
    if n <= n_ref:
      return (0.5 / (n_ref - n_min)) * (n - n_min)
    return (0.5 / (n_max - n_ref)) * (n - n_ref) + 0.5
  


  def _compute_rastrigin_value(self, x: npt.NDArray[np.float64]) -> float:
    d = x.shape[0]
    rastrigin = lambda x_i: x_i ** 2 - 10.0 * math.cos(2.0 * math.pi * x_i)
    sigma = np.sum(np.vectorize(rastrigin)(x))
    return -10.0 * d - sigma # queremos el valor negativo
