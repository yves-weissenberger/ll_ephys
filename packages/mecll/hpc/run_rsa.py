import numpy as np
from typing import List

def get_spatial_distance_matrix(poke_pos: List[List[int, int]], seq: List[int]) -> np.ndarray:

    pp = [pos for i,pos in enumerate(poke_pos) if i in seq]

    return None
    
    
def remove_diagonal(A: np.ndarray) -> np.ndarray:
    """ Useful for when dealing with correlation or other distance matrices """
    removed = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], int(A.shape[0])-1, -1)
    return np.squeeze(removed)


def line_distance_matrix(seq: List[int],p=1) -> np.ndarray:
    """Given a sequence of states, get the distance in task space
       between pokes retunging the results ordered by spatial position

    Args:
        seq (List[int]): the sequence
        p (int, optional): raise distance to power p. Defaults to 1.

    Returns:
        np.ndarray: The task distance matrix
    """
    seq = list(seq)
    n = len(seq)
    d = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            ind1 = seq.index(i)
            ind2 = seq.index(j)
            d[i,j] = abs(ind1-ind2)**p
    return d