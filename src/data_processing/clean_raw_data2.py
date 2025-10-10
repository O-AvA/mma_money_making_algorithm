"""
Data Cleaning Module for ufcstats.com data scraped by Greco1899's algorithm. 
https://github.com/Greco1899

Key functionality:
- Combine the 4 raw datasets into a single cleaned dataset, ready for further feature engineering. 
- Engineer basic features such as date, height, age, bout format etc..  
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import mlflow

#from config import settings
from src.utils.general import (
    open_csv, store_csv, creaopen_file, get_data_path, ensure_dir,
    get_best_match, stat_names)

class UFCDataProcessor:
    """
    Process data from ufcstats.com scraped by Greco1899's algorithm.
    https://github.com/Greco1899/scrape_ufc_stats.  
    Create basic features and prepares data for further feature engineering.


    """
    
    def __init__(self, prefer_external=True, new_fights_only=False): 
        """
        Initialize the data processor.

        Params: 
            prefer_external bool: 
                whether to take (probably) updated data from Graeco1899's ufc_stats_scraper repository 
                or from own data folder.  

            new_fights_only bool: WIP!!! Dont set to True  
                default is False. If True, only processes new samples in 
                stead of the entire dataset. 

                I am not actually sure if I would suggest this since Catch weight bout don't mention 
                sex, so the algorithm infers sex from fighter's other non-Catch weight bouts (if any). 
                The algorithm is relatively fast anyway so it doesn't matter that much. 
        """
        if new_fights_only: 
            raise Exception('Sorry, WIP. Set new_fights_only to False') 
    
        self.raw_data_path = get_data_path("raw")
        self.interim_data_path = get_data_path("interim")
        self.processed_data_path = get_data_path("processed") # Hebben we deze nodig? 
        self.interim_path = get_data_path("interim") 

        # Params 
        self.prefer_external = prefer_external
        self.new_fights_only = new_fights_only 
        
        # Ensure output directories exist
        ensure_dir(self.raw_data_path) 
        ensure_dir(self.interim_data_path)

        # Initialize attributes for DataFrames
        self.events = None
        self.results = None
        self.stats = None
        self.fighters = None
        self.alt_spellings = None 
        
        logger.info("Initialized UFC data processor")

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all necessary raw data files, either externally or internally. Also saves external data. 

        """
        raw_files = {
            'df_events': "ufc_event_details.csv",
            'df_outcomes': "ufc_fight_results.csv", 
            'df_stats': "ufc_fight_stats.csv",
            'df_fighters': "ufc_fighter_tott.csv"
        } 
        scrape_ufc_stats_main = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
      
        for name, file_name in raw_files.items():
            ufc_stats_scraper_data_link = f'{scrape_ufc_stats_main}/{file_name}'

            if self.prefer_external: 
                try: 
                    df = pd.read_csv(ufc_stats_scraper_data_link)  

                    logger.info('Succesfully loaded data from https://github.com/Greco1899/scrape_ufc_stats')

                    # If new_fights_only = True, only processes new fights. I think it's best to do exclude 
                    # df_fighters from this in case fighter height and stats get updated, such that drop_duplicates() 
                    # doens't work.

                    df_new = df.copy()  
                    if name != 'df_fighters':
                        if self.new_fights_only: 
                            df_old = open_csv(self.raw_data_path / file_name)
                            if len(df_new) > len(df_old): 
                                df_new = pd.concat([df_new, df_old]).drop_duplicates() 
                                logger.info(f'Dropped {len(df)} columns from {file_name}.')  
                            else: 
                                raise Exception(f'Dataset {file_name} appears to not yet have been updated')

                    store_csv(df, self.raw_data_path / file_name) 
                except Exception as e: 
                    logger.info(f'Could not fetch data from ufc_stats_scraper: {e}') 
                    logger.info('Loading from data instead.') 
                    df_new = pd.read_csv(self.raw_data_path / file_name)
            else: 
                df_new = pd.read_csv(self.raw_data_path / file_name)

            setattr(self, name, df_new)

        # Additionally, we need to load or initialize the alternative spellings dataframe. 
        self.df_alt_spellings = creaopen_file(self.interim_data_path / 'alternative_spellings_internal.csv') 
        

    def clean_data(self):
        """
        Cleans and processes the four datasets from ufcstats
        
        Returns:
            Cleaned fight results DataFrame
        """
        pass