import pandas as pd
import numpy as np
from ..SVD_analysis import variance_explained_U,variance_explained_V,variance_explained_both


""" What we want to is to calculate the fraction of variance explained across cross-validated
    halves of the data in the SVD analysis. """ 


def get_split_data_halves():
    """ Here we want to split data in half so as to have parts for testing how well
        SVD explains the data across vs within task. 

        Even in the line want to do this modulo the length of the line
    """
    return None

def sort_according_to_space():
    """
    
    
    """
    return None

def 
if __name__=="__main__":

    df = pd.read_csv('ses')
    for session in session_df.iterrows():
        out = get_split_data_halves()
        task1_p1, task1_p2, task2_p1, task2_p2 = out