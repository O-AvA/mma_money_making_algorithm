"""
General functions 
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from loguru import logger

from fuzzywuzzy import process, fuzz


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_data_path(subfolder: str = None) -> Path:
    """
    Get path to data directory with optional subfolder.
    
    Args:
        subfolder: Optional subfolder within data directory
        
    Returns:
        Path to data directory or subfolder
    """
    data_path = get_project_root() / "data"
    
    if subfolder:
        data_path = data_path / subfolder
        ensure_dir(data_path)
    
    return data_path

def get_src_path(subfolder: str = None) -> Path:
    """
    Get path to data directory with optional subfolder.
    
    Args:
        subfolder: Optional subfolder within data directory
        
    Returns:
        Path to data directory or subfolder
    """
    data_path = get_project_root() / "src"
    
    if subfolder:
        data_path = data_path / subfolder
        ensure_dir(data_path)
    
    return data_path

def get_best_match(name, choices, threshold=90):
    # Find the closest match using fuzzywuzzy
    match, score = process.extractOne(name, choices, scorer=fuzz.ratio)
    if score >= threshold:
        return match, score  
    else:
        return name, 0 



def open_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Open CSV file with error handling.
    
    Args:
        file_path: Path to CSV file (relative to project root or absolute)
        **kwargs: Additional arguments passed to pd.read_csv()
        
    Returns:
        DataFrame with CSV data
    """
    # Convert to Path object
    path = Path(file_path)
    
    # If not absolute, make it relative to project root
    if not path.is_absolute():
        path = get_project_root() / path
    
    try:
        df = pd.read_csv(path, **kwargs)
        logger.debug(f"Loaded CSV: {path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {e}")
        return pd.DataFrame()




def store_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Store DataFrame as CSV with error handling.
    
    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
        **kwargs: Additional arguments passed to df.to_csv()
    """
    # Convert to Path object
    path = Path(file_path)
    
    # If not absolute, make it relative to project root
    if not path.is_absolute():
        path = get_project_root() / path
    
    # Ensure directory exists
    ensure_dir(path.parent)
    
    try:
        # Default arguments
        csv_kwargs = {'index': False}
        csv_kwargs.update(kwargs)
        
        df.to_csv(path, **csv_kwargs)
        logger.debug(f"Saved CSV: {path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error saving CSV {path}: {e}")
        raise


def creaopen_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Create file if it doesn't exist, otherwise open it.
    
    Args:
        file_path: Path to file
        
    Returns:
        DataFrame (empty if file doesn't exist)
    """
    path = Path(file_path)
    
    if not path.is_absolute():
        path = get_project_root() / path
    
    if path.exists():
        return open_csv(path)
    else:
        logger.debug(f"File doesn't exist, returning empty DataFrame: {path}")
        return pd.DataFrame()



def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
    """
    path = Path(file_path)
    
    if not path.is_absolute():
        path = get_project_root() / path
    
    ensure_dir(path.parent)
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved JSON: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON {path}: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with JSON data
    """
    path = Path(file_path)
    
    if not path.is_absolute():
        path = get_project_root() / path
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON: {path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading JSON {path}: {e}")
        return {}


# Feature engineering utilities
def stat_names(which=['','l','A','D'],
               rounds=True,
               fighters=True,
               extras=True,
               stds=False,
               avgs=False) -> List: 
    """
    Gives string of fight stat column names in clean_ufcstats-com_data.csv. "

    Args: 
        list which: 
            '' = Attempted (missed + landed) 
            'l' = landed
            'A' = accuracy  
            'D' = defense accuracy
        bool rounds: 
            if True, include round label
        bool or list(int) fighters: 
            if True, include fighter labels for both fighters 
            or provide list(int) with 1, 2 or both 
        bool extras: 
            if True, include CT, SA, KD
        bool avg: 
            if True, include avg, std. Incompatible with rounds = True

    Returns: 
        list(str) column names 
    """

    # Returns a list of column names of processed fights 
    stat_refs = ['SS','TS','TD','HS','BS','LS','DS','CS','GS']
    stat_names = []
    for suff in which: 
        stat_names += [stat_ref+suff for stat_ref in stat_refs]
    if extras: 
        stat_names += ['CT','SA','KD']

    def add_rounds(stats): 
        stat_names_r = [] 
        for stat_name in stats: 
            stat_names_r += [f'{stat_name}r{ri}' for ri in range(1,6)]
        return stat_names_r 

    def add_fighter_label(stats): 
        if isinstance(fighters, bool): 
            flabels = ['f1','f2'] if fighters else [] 
        else: 
            flabels = [f'f{fi}' for fi in fighters] 
        if len(flabels)> 0: 
            stat_names_f = [] 
            for stat_name in stats: 
                stat_names_f.extend([stat_name+flabel for flabel in flabels]) 
        else: 
            stat_names_f = stats
        return stat_names_f 

    def add_avg(stats): 
        return [stat+'avg' for stat in stats] 

    def add_std(stats): 
        return [stat+'std' for stat in stats] 
    
    stats_r = add_rounds(stat_names) if rounds else [] 
    stats_rf = add_fighter_label(stats_r)
    
    stats_a = add_avg(stat_names) if avgs else [] 
    stats_s = add_std(stat_names) if stds else [] 
    stats_as = stats_a + stats_s  
    stats_abf = add_fighter_label(stats_as)

    some_fighters = True if (fighters or (isinstance(fighters, list) and len(fighters) > 0)) else False  
    stat_names = stats_rf + stats_abf if (rounds or  some_fighters) else stat_names 

    # Align
    both_fighters = True if (isinstance(fighters, list) and len(fighters)==2) else False 
    if both_fighters: 
        snames1 = [stat for stat in stat_names if stat[-2:] == 'f1'] 
        snames2 = [stat.replace('f1', 'f2') for stat in snames1] 
        stat_names = snames1 + snames2 

    return stat_names 

def expectancy(rating1: float, rating2: float, c: float = 200.0) -> float:
    """
    Calculate ELO expectancy (probability of win for fighter 1).
    
    Args:
        rating1: ELO rating of fighter 1
        rating2: ELO rating of fighter 2
        c: ELO constant
        
    Returns:
        Expected probability of win for fighter 1
    """
    return 1 / (1 + 10**((rating2 - rating1) / c))


def compute_elo(current_rating: float, 
                expected_score: float, 
                actual_score: float, 
                K: float) -> float:
    """
    Compute new ELO rating after a fight.
    
    Args:
        current_rating: Current ELO rating
        actual_score: 
        expected_score:  
        K: K factor 
        
    Returns:
        New ELO rating
    """
    # Update rating
    if isinstance(current_rating, pd.Series): 
        current_rating = current_rating.to_numpy() 
    if isinstance(actual_score, pd.Series): 
        actual_score = actual_score.to_numpy() 
    if isinstance(expected_score, pd.Series): 
        expected_score = expected_score.to_numpy() 
    if isinstance(K, pd.Series): 
        K = K.to_numpy() 

    new_rating = current_rating + K * (actual_score - expected_score)

    if isinstance(current_rating, np.ndarray): 
        current_rating = pd.Series(current_rating)
        new_rating = pd.Series(new_rating)
    if isinstance(actual_score, np.ndarray): 
        actual_score = pd.Series(actual_score) 
    if isinstance(expected_score, np.ndarray): 
        expected_score = pd.Series(expected_score)
    if isinstance(K, np.ndarray): 
        K = pd.Series(K)  
    
    return new_rating


# Data manipulation utilities
class Dict(dict):
    def append_stat(self,key,val):
        if key in self: 
            if isinstance(val, list): 
                self[key].extend(val) 
            else: 
                self[key].append(val) 
        else:
            if isinstance(val, list): 
                self[key] = val 
            else: 
                self[key] = [val] 



