from typing import List, Optional


# external modules
import numpy as np
from scipy.spatial import distance_matrix

def get_spatial_distance_matrix(poke_pos: List[List[int]], seq: List[int], p: int =1) -> np.ndarray:
    """Get physical distance matrix

    Args:
        poke_pos (List[List[int, int]]): Poke positions
        seq (List[int]): sequence that pokes are visited in the task
        p (int, optional): distance. Defaults to 1.

    Returns:
        np.ndarray: [description]
    """
    pp = [pos for i,pos in enumerate(poke_pos) if i in seq]

    return distance_matrix(pp,pp,p=p)


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
    sorted_seq  = sorted(seq)
    n = len(seq)
    d = np.zeros([n,n])
    for iix,i in enumerate(sorted_seq):
        for jix,j in enumerate(sorted_seq):
            ind1 = seq.index(i)
            ind2 = seq.index(j)
            d[iix,jix] = abs(ind1-ind2)**p
    return d


def remove_diagonal(A: np.ndarray) -> np.ndarray:
    """ Useful for when dealing with correlation or other distance matrices """
    removed = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], int(A.shape[0])-1, -1)
    return np.squeeze(removed)