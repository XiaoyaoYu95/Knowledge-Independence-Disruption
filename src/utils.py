# utils.py
import numpy as np
import pandas as pd
from scipy import stats

def read_big_csv(filename, sep=',', header='infer', names=None, 
                index_col=None, usecols=None, dtype=None, chunksize=1024*1024, 
                parse_dates=False, nrows=None, compression=None):
    """
    Read a large CSV file in chunks and concatenate the results into a single DataFrame.
    Parameters:
    - filename: str, path to the CSV file.
    - sep: str, delimiter to use (default is ',').
    - header: int or list of int, row(s) to use as the column names (default is 'infer').
    - names: list of str, column names to use (default is None).
    - index_col: int or str, column to set as index (default is None).
    - usecols: list of str or int, columns to read (default is None).
    - dtype: dict, data types for columns (default is None).
    - chunksize: int, number of rows per chunk (default is 1024*1024).
    - parse_dates: bool or list, whether to parse dates (default is False).
    - nrows: int, number of rows to read (default is None, meaning all rows).
    - compression: str, type of compression (default is None).
    Returns:
    - res_df: pandas DataFrame, concatenated DataFrame from all chunks.
    """
    df_chunk = pd.read_csv(filename, sep=sep, header=header, names=names, 
                       index_col=index_col, usecols=usecols, dtype=dtype, chunksize=chunksize, 
                       parse_dates=parse_dates, nrows=nrows, compression=compression)
    res_chunk=[]
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df=pd.concat(res_chunk)
    return res_df


def dataframe_nor(df): 
    """
    standardizing dataframe by z-score
    """
    # select numeric columns and apply z-score normalization
    df_nor = df.select_dtypes(include=[np.number]).apply(stats.zscore)
    # replace the original numeric columns with standardized values
    for col in df_nor.columns:
        df[col] = df_nor[col]
    return df
    
    
def load_data(file_path, data_type, all_vars):
    """
    Load data from a pickle file and merge with additional dataframes.
    Args:
        file_path (str): Path to the pickle file.
        data_type (str): Type of data processing ('nor' for normalization).
        all_vars (list): List of variables to keep in the dataframe.
    Returns:
        pd.DataFrame: Processed dataframe with specified variables.
    """
    df = pd.read_pickle(file_path)

    # transform 'C5' column to log scale
    df['log_C5'] = df['C5'].apply(np.log1p)# 对'C5'列做log(x + 1)
    df['Citation_percentile'] = df.groupby(['Year', 'Field'])['Copen'].transform(rank) # calculate the citation percentile within each year and field

    # Ensure all specified variables are in the dataframe
    df = df[all_vars]
    df.dropna(axis=0,how='any',inplace=True)
    
    if data_type == "nor":# standardizing dataframe
        df = dataframe_nor(df)

    # add a unique identifier for each paper
    df['paper_id'] = list(range(len(df)))
    
    return df

    
def rank(J_):
    """
    Rank the values in J_ and return a list of ranks in percentiles.
    We record the highest rank of each unique value in J_ for which appears multiple times.
    """
    J_sorted = sorted(J_)
    N = len(J_)
    J_rank_dict = {}
    for i in range(N):
        J_rank_dict[J_sorted[i]] = (i+1)*100/N
    
    J_rank_list = []
    for j in J_:
        J_rank_list.append(J_rank_dict[j])

    return J_rank_list


def rank_bin(J_,bin_list,label_list):
    """
    Rank the values in J_ and return a binned rank list based on predefined bins.
    """
    J_rank_list = rank(J_)
    J_rank_bin = pd.cut(x = J_rank_list, bins = bin_list, labels = label_list, include_lowest = True)
    return J_rank_bin
    

def binary_check(D_list):
    """
    Check if the values in D_list are greater than 0 and return a binary list.
    """
    D_tag = [0]*len(D_list)
    for i in range(len(D_list)):
        if D_list[i] > 0:
            D_tag[i] = 1
    return D_tag


def convert_to_decade(year):
    """
    Convert a year to its corresponding decade.
    For example, 1987 becomes '1980s', 2001 becomes '2000s'.
    """
    if type(year) is not int:
        decade = None
    else:
        decade = (year // 10) * 10
    return f"{decade}s"