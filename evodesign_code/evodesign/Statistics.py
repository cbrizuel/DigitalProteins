import evodesign.Sequence as Sequence
from itertools import combinations
import pandas as pd
import numpy as np





def average_amino_acid_loss(population: pd.DataFrame) -> float:
    """
    Computes the average number of amino acid letters lost for each residue 
    position. An amino acid letter is considered lost when none of the
    sequences in the given population contain such letter in the corresponding
    residue position. 
    
    This value is a way of measuring diversity loss accross the length of 
    the sequences in the given population; the closer this value is to 20 
    (the number of essential amino acids), the less diversity the population 
    has.

    Parameters
    ----------
    population : pandas.DataFrame
        The population for which the average amount of missing residues will
        be computed.

    Returns
    -------
    float
        The average amount of missing residues.
    """
    m = len(Sequence.AMINO_ACIDS)
    n = len(population.iloc[0]['sequence'])
    data = np.array([
        m - len({ seq[i] for _, seq in population['sequence'].items() })
        for i in range(n)
    ])
    return data.mean()
  


def average_sequence_identity(population: pd.DataFrame) -> float:
    """
    Computes the average sequence identity for each sequence in the given 
    population by comparing it against all other sequences in the population.
    Then, returns the weighted average of the obtained averages. 
    
    This value is a way of measuring diversity loss accross all sequences in
    the given population; to closer this value is to the sequence length,
    the less diversity the population has.

    Parameters
    ----------
    population : pandas.DataFrame
        The population for which the average sequence identity will be computed.

    Returns
    -------
    float
        The average sequence identity.
    """
    # for each sequence in the population, compute its identity against all
    # other sequences; take care not to compare the same pair of sequences
    # more than once
    k = None
    identities = {}
    for i, j in combinations(range(len(population)), 2):
        if i != k:
            if k in identities:
                identities[k] = np.array(identities[k])
            k = i
            identities[k] = []
        a = population.iloc[i]['sequence']
        b = population.iloc[j]['sequence']
        identities[k].append(sum(c == d for c, d in zip(a, b)))
    identities[k] = np.array(identities[k])
    # compute the average identity for each sequence in the population;
    # then, since every sequence is compared against a different number of
    # sequences, compute the weighted average of the resulting averages
    averages = np.array([ data.mean() for _, data in identities.items() ])
    weights = np.array([ len(data) for _, data in identities.items() ])
    return np.average(averages, weights=weights)
  