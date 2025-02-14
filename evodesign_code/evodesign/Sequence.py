import numpy as np
from typing import Optional, Dict, List



AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')





def create_random(rng: np.random.Generator, 
                  length: int,
                  allowed_letters: Optional[Dict[int, List[str]]] = None
                  ) -> str:
    """
    Generates a random sequence of amino acids with uniform probability for
    all amino acids at each residue position. 

    Parameters
    ----------
    rng : numpy.random.Generator
        The RNG to be used to choose the letters of the produced sequence.
    length : int
        The number of residues in the produced sequence.
    allowed_letters : Dict[int, List[str]], optional
        A description of which letters are allowed to be chosen for certain positions
        in the sequence. If no letter pool is specified for a given position, then no
        restrictions in the letter selection will be imposed at that position. Default
        is `None`, which means that any amino acid letter can be chosen at any position.

    Returns
    -------
    str
        The generated sequence. Each residue is represented as a
        single letter corresponding to one of the 20 standard amino acids.
    """
    if allowed_letters is None:
        return ''.join(rng.choice(AMINO_ACIDS, size=length).tolist())
    allowed_letters = allowed_letters_desc(length, allowed_letters)
    return ''.join([
        rng.choice(letter_pool)
        for i, letter_pool in allowed_letters.items()
    ])



def swap_letter(rng: np.random.Generator, 
                letter: str,
                allowed_letters: List[str] = AMINO_ACIDS
                ) -> str:
    """
    Produce a single random amino acid letter that's different from the one
    specified by `letter`.

    Parameters
    ----------
    rng : numpy.random.Generator
        The RNG to be used to produce the result of the coin toss.
    letter : str
        A single letter representing one of the 20 standard amino acids.
    allowed_letters : List[str], optional
        The set of letters from which the new letter will be chosen. By default,
        this pool is composed of all 20 letters representing the standard amino acids. 

    Returns
    -------
    str
        A single letter that is different from the one specified by `letter`. 
        The produced letter also corresponds to one of the 20 standard amino 
        acids.
    """
    if len(allowed_letters) < 2:
        return letter
    new_letter = letter
    while new_letter == letter:
        new_letter = allowed_letters[rng.choice(len(allowed_letters))]
    return new_letter



def allowed_letters_desc(length: int, 
                         desc: Optional[Dict[int, List[str]]] = None
                         ) -> Dict[int, List[str]]:
    """
    Creates a new description object for the allowed amino acid letters at each position
    of the designed sequences. 

    If an already existing description is provided, then positions that where not
    explicitly given any restrictions in said description will be filled with the 
    default restrictions--that is, allowing all 20 standard amino acid letters at that 
    position.

    If no description object is provided, then the description object returned by
    this function will allow all 20 standard amino acid letters at every position.

    Parameters
    ----------
    length : int
        The number of amino acids in the designed sequences.
    desc : Dict[int, List[str]], optional
        An already existing description object which missing position descriptions
        will be filled. If none is provided, then all 20 standard amino acids will be
        allowed at every position.

    Returns
    -------
    Dict[int, List[str]]
        The newly created description object
    """
    if desc is None:
        return { i: AMINO_ACIDS for i in range(length) }
    if len(list(desc.keys())) >= length:
        return desc
    return {
        i: desc[i] if i in desc else AMINO_ACIDS
        for i in range(length)
    }
