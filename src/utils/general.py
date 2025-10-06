"""
General utility functions for UFC ML Betting System

This module contains common utility functions used throughout the project.
Migrated and modernized from _old_ufcbets/general.py

Key functionality:
- File I/O operations (CSV reading/writing)
- Data manipulation helpers
- ELO calculation utilities
- Feature engineering helpers
- Statistical functions
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

def get_elo_feature_names(which=["",'l','D','A'], fighters=[1,2], 
                          avgs=True, rounds=False, include_final=False, extras=False): 
    snames = stat_names(which=which, fighters=fighters, rounds=rounds, avgs=avgs, extras=False)

    elo_names = [] 
    for fi in fighters: 
        elo_names.extend([sname.replace(f'f{fi}',f'Rf{fi}') for sname in snames]) 
        elo_names.extend([sname.replace(f'f{fi}',f'Ef{fi}') for sname in snames]) 
        if include_final: 
            elo_names.extend([sname.replace(f'f{fi}',f'_REMOVE_Rf{fi}') for sname in snames]) 
    return elo_names  


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


def count_cols(data: List[List], n: int = 0, rank: bool = True) -> Dict[str, List[int]]:
    """
    Count occurrences of elements in nested lists.
    
    Args:
        data: Nested list of elements
        n: Minimum count threshold
        rank: Whether to rank by count
        
    Returns:
        Dictionary with element counts
    """
    # Flatten the data
    flattened = np.array(data, dtype=object).ravel()
    unique_elements, counts = np.unique(flattened, return_counts=True)
    
    # Filter by threshold
    mask = counts > n
    filtered_elements = unique_elements[mask]
    filtered_counts = counts[mask]
    
    ranked_cols = {}
    
    if rank:
        # Sort by count descending
        sorted_indices = np.argsort(-filtered_counts)
        for i in sorted_indices:
            ranked_cols[filtered_elements[i]] = [filtered_counts[i]]
    
    return ranked_cols


# Fighter data utilities
def get_dfodfe(name: str, tau: int = -1, current_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get fighter statistics and ELO data.
    
    Args:
        name: Fighter name
        tau: Time index filter (-1 for all)
        current_only: Whether to get only current tau
        
    Returns:
        Tuple of (stats_df, elo_df)
    """
    # Load fighter statistics
    stats_path = get_data_path("interim/fighters") / f"{name}_stats.csv"
    elo_path = get_data_path("interim/fighters_clean") / f"{name}.csv"
    
    dfo = open_csv(stats_path)
    dfe = open_csv(elo_path)
    
    # Apply tau filtering
    if tau >= 0:
        if current_only:
            dfo = dfo[dfo['tau'] == tau] if 'tau' in dfo.columns else dfo
            dfe = dfe[dfe['tau'] == tau] if 'tau' in dfe.columns else dfe
        else:
            dfo = dfo[dfo['tau'] <= tau] if 'tau' in dfo.columns else dfo
            dfe = dfe[dfe['tau'] <= tau] if 'tau' in dfe.columns else dfe
    
    # Sort by tau
    if 'tau' in dfo.columns:
        dfo = dfo.sort_values(by='tau').reset_index(drop=True)
    if 'tau' in dfe.columns:
        dfe = dfe.sort_values(by='tau').reset_index(drop=True)
    
    return dfo, dfe



### FROm here you may be able to delete 

def append_fighter_data(fighter_stats: Dict[str, List]) -> None:
    """
    Append fighter statistics to individual fighter files.
    
    Args:
        fighter_stats: Dictionary with fighter statistics
    """
    if not fighter_stats:
        return
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(fighter_stats)
    
    if 'name' not in stats_df.columns:
        logger.error("Fighter stats missing 'name' column")
        return
    
    # Process each fighter
    for fighter in stats_df['name'].unique():
        fighter_data = stats_df[stats_df['name'] == fighter].sort_values(by='tau').reset_index(drop=True)
        
        # Load existing fighter data
        fighter_file = get_data_path("interim/fighters_clean") / f"{fighter}.csv"
        existing_df = create_or_open_file(fighter_file)
        
        # Remove overlapping columns (except tau and name)
        cols_to_drop = [col for col in fighter_data.columns 
                       if col != 'tau' and col != 'name' and col in existing_df.columns]
        if cols_to_drop:
            existing_df = existing_df.drop(columns=cols_to_drop)
        
        # Concatenate data
        if not existing_df.empty:
            # Sort existing data
            if 'tau' in existing_df.columns:
                existing_df = existing_df.sort_values(by='tau').reset_index(drop=True)
            
            # Combine datasets
            combined_df = pd.concat([existing_df, fighter_data.drop(columns=['name', 'tau'])], 
                                   axis=1, ignore_index=False)
        else:
            combined_df = fighter_data
        
        # Save updated data
        # Only save columns from 'name' onwards (remove any unnamed columns)
        if 'name' in combined_df.columns:
            name_idx = combined_df.columns.get_loc('name')
            combined_df = combined_df.iloc[:, name_idx:]
        
        store_csv(combined_df, fighter_file)
    
    logger.info(f"Updated data for {len(stats_df['name'].unique())} fighters")


def get_keys(round_avg: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """
    Get feature column names for model training.
    
    Args:
        round_avg: Whether to include round-averaged features
        
    Returns:
        Tuple of (elo_ref, elo_strings, all_keys)
    """
    elo_ref = ELO_TYPES.copy()
    elo_strings = ELO_TYPES.copy()
    
    # Add ELO variants
    for elo_type in elo_ref:
        for variant in ELO_VARIANTS[1:]:
            elo_strings.append(elo_type + variant)
    
    # Build all feature keys
    keys = ['elo1']
    
    for fighter_idx in range(2):  # Fighter 1 and Fighter 2
        for elo_type in elo_strings:
            if round_avg:
                # Round-specific features
                for round_num in range(1, 6):  # Rounds 1-5
                    keys.append(f"{elo_type}{fighter_idx + 1}r{round_num}")
                    keys.append(f"CT{fighter_idx + 1}r{round_num}")  # Control time
                    keys.append(f"SA{fighter_idx + 1}r{round_num}")  # Strikes attempted
                    keys.append(f"SDA{fighter_idx + 1}r{round_num}")  # Strikes absorbed
                    keys.append(f"REV{fighter_idx + 1}r{round_num}")  # Reversals
                    keys.append(f"KD{fighter_idx + 1}r{round_num}")  # Knockdowns
            else:
                # Fight-level features
                keys.append(f"{elo_type}{fighter_idx + 1}")
                keys.append(f"CT{fighter_idx + 1}")
                keys.append(f"SA{fighter_idx + 1}")
                keys.append(f"SDA{fighter_idx + 1}")
                keys.append(f"REV{fighter_idx + 1}")
                keys.append(f"KD{fighter_idx + 1}")
        
        if fighter_idx == 0:
            keys.append('elo2')
    
    return elo_ref, elo_strings, keys


def show_cols(dataframe: pd.DataFrame) -> None:
    """
    Display DataFrame columns with indices.
    
    Args:
        dataframe: DataFrame to inspect
    """
    for idx, col in enumerate(dataframe.columns):
        print(f"{idx}: {col}")


def compare_string(str1: str, str2: str) -> None:
    """
    Compare two strings character by character for debugging.
    
    Args:
        str1: First string
        str2: Second string
    """
    l1, l2 = len(str1), len(str2)
    print('Strings are equal:', str1 == str2)
    
    if l1 > l2:
        print("string 1 longer")
    elif l2 > l1:
        print("string 2 longer")
    else:
        print("same size")
    
    for i in range(min(len(str1), len(str2))):
        print(f"'{str1[i]}' == '{str2[i]}'")


def open_raw(file_number: int) -> pd.DataFrame:
    """
    Open raw data files by index.
    
    Args:
        file_number: Index of file to open (0-4)
        
    Returns:
        DataFrame with raw data
    """
    raw_files = [
        "data/raw/fights/ufc_event_details.csv",
        "data/raw/fights/ufc_fight_results.csv",
        "data/raw/fights/ufc_fight_stats.csv",
        "data/raw/fights/ufc_fighter_tott.csv",
        "data/raw/alt_spellings.txt"
    ]
    
    if file_number < 0 or file_number >= len(raw_files):
        logger.error(f"Invalid file number: {file_number}")
        return pd.DataFrame()
    
    file_path = raw_files[file_number]
    
    if file_number == 4:  # alt_spellings.txt
        return pd.read_csv(file_path, header=None, names=['dfo name', 'dff name'])
    else:
        return open_csv(file_path)


# Prediction and confidence analysis utilities
def get_confs(probs: np.ndarray, 
              names: List[str], 
              opponents: List[str], 
              save: bool = True) -> Optional[Dict[str, float]]:
    """
    Calculate fighter confidence scores from predictions.
    
    Args:
        probs: Prediction probabilities [n_fights, 3]
        names: Fighter names
        opponents: Opponent names
        save: Whether to save to trading/preds.csv
        
    Returns:
        Dictionary of fighter confidence scores (if save=False)
    """
    conf_dict = {}
    
    for i in range(len(probs)):
        # Fighter 1 confidence: P(win) + 0.5 * P(draw)
        conf_dict = append_stat(conf_dict, names[i], probs[i][0] + probs[i][1] / 2)
        # Fighter 2 confidence: P(loss) + 0.5 * P(draw)
        conf_dict = append_stat(conf_dict, opponents[i], probs[i][2] + probs[i][1] / 2)
    
    # Average confidences for fighters with multiple fights
    conf_dict = {k: np.nanmean(conf_dict[k]) for k in conf_dict.keys()}
    
    if save:
        # Save to trading predictions file
        conf_dict_lists = {key: [conf_dict[key]] for key in conf_dict.keys()}
        trading_file = get_data_path("interim") / "trading" / "preds.csv"
        
        existing_df = create_or_open_file(trading_file)
        new_df = pd.DataFrame(conf_dict_lists)
        
        combined_df = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
        # Remove unnamed columns
        combined_df = combined_df[[col for col in combined_df.columns if 'Unnamed' not in col]]
        
        store_csv(combined_df, trading_file)
        
        return None
    else:
        return conf_dict


def analyze_confidence(probs: np.ndarray,
                       names: List[str],
                       opponents: List[str],
                       results: List[int],
                       granularity: int = 3) -> None:
    """
    Analyze prediction confidence vs actual accuracy.
    
    Args:
        probs: Prediction probabilities
        names: Fighter names
        opponents: Opponent names  
        results: Actual results (0=fighter1 win, 2=fighter2 win)
        granularity: Confidence bucket granularity
    """
    # Initialize confidence buckets
    corrs = {str(x): 0 for x in range(0, 100, granularity)}
    counts = {str(x): 0 for x in range(0, 100, granularity)}
    
    for i, result in enumerate(results):
        # Get confidence as percentage
        conf_pct = round(100 * (probs[i][0] + probs[i][1] / 2))
        
        if conf_pct > 100 - conf_pct:  # Predict fighter 1 win
            rest = conf_pct % granularity
            bucket = conf_pct - rest
            counts[str(bucket)] += 1
            if result == 0:  # Fighter 1 actually won
                corrs[str(bucket)] += 1
        else:  # Predict fighter 2 win
            rest = (100 - conf_pct) % granularity
            bucket = 100 - conf_pct - rest
            counts[str(bucket)] += 1
            if result == 2:  # Fighter 2 actually won
                corrs[str(bucket)] += 1
    
    # Calculate accuracy by confidence bucket
    accuracy_dict = {}
    for key in corrs.keys():
        if counts[key] > 0:
            accuracy_dict[key] = [corrs[key] / counts[key]]
        else:
            accuracy_dict[key] = [0]
    
    # Save results
    conf_file = get_data_path("interim") / "trading" / f"confacc_{granularity}.csv"
    existing_df = create_or_open_file(conf_file)
    new_df = pd.DataFrame(accuracy_dict)
    
    combined_df = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
    combined_df = combined_df.drop(columns=[col for col in combined_df.columns if 'Unnamed' in col])
    
    store_csv(combined_df, conf_file)
    
    logger.info(f"Confidence analysis saved to {conf_file}")
