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
        logger.info("Cleaning fight results data")

        dfo = self.df_outcomes.copy() 
        dff = self.df_fighters.copy()
        dfs = self.df_stats.copy()
        dfe = self.df_events.copy()
        dft = self.df_alt_spellings.copy()
        
        # PREPARE DATA 
        logger.info('Preparing raw data sets...') 
        # First add Alexandre Topuria cause its missing from ufc_fighter_tott.csv 
        # for some reason
        dff['FIGHTER'] = dff['FIGHTER'].str.strip()
        if 'Alexandre Topuria' not in dff['FIGHTER'].values: 
            new_fighter = { 
                        'FIGHTER': 'Alexandre Topuria', 
                        'HEIGHT': "5'" + ' 8"',   
                        'WEIGHT': "135 lbs.", 
                        'REACH': '68"',   
                        'STANCE': np.nan,  
                        'DOB': "Jan 28, 1996",  
                        'URL': np.nan
            }
            new_fighter = {key: [new_fighter[key]] for key in new_fighter.keys()} 
            dff = pd.concat([dff,pd.DataFrame(new_fighter)],axis=0,ignore_index=True)

        # The earliest fight in the dataset will be
        # UFC 31: Locked and Loaded, May 04, 2001. 
        # This is the first event to use the unified rules of MMA 
        # and to not include a 2 rounds time format.
        dfo['EVENT'] = dfo['EVENT'].str.strip()
        dfs['EVENT'] = dfs['EVENT'].str.strip() 
        dfe['EVENT'] = dfe['EVENT'].str.strip()

        if not self.new_fights_only: 
            ff = dfo[dfo['EVENT']=='UFC 31: Locked and Loaded'].index.values[-1] + 1
            include_fights = ff  # ff means all fights in the datasets. x < ff is the first x fights. 
            dfo = dfo.iloc[ff-include_fights:ff,:]
            dfs = dfs[dfs['EVENT'].isin(dfo['EVENT'])] 
            dfe = dfe[dfe['EVENT'].isin(dfo['EVENT'])] 

        logger.info('Matching bouts and fighters...')
        # Matching datasets using bout and event -> f_id   
        # Basic cleaning and getting fighter names
        dfo['BOUT'] = dfo['BOUT'].str.replace('  ',' ').str.strip()
        dfs['BOUT'] = dfs['BOUT'].str.replace('  ',' ').str.strip()
        dfo[['name f1', 'name f2']] = dfo['BOUT'].str.split(' vs. ', expand=True)
        dfo['name f1'] = dfo['name f1'].str.strip() 
        dfo['name f2'] = dfo['name f2'].str.strip()
        dft['dfo name'] = dft['dfo name'].str.strip()
        dft['dff name'] = dft['dff name'].str.strip()
            
        # Making sure each each fighter has only 1 name using some 
        # partially manually compiled dataset and fuzzywuzzy further down.
        alt_map = dict(zip(dft['dff name'], dft['dfo name']))
        dfo['name f1'] = dfo['name f1'].map(alt_map).fillna(dfo['name f1'])
        dfo['name f2'] = dfo['name f2'].map(alt_map).fillna(dfo['name f2'])
        dff['FIGHTER'] = dff['FIGHTER'].map(alt_map).fillna(dff['FIGHTER'])

        del alt_map

        ### INITIAL WEIGHTCLASS PROCESSING AND HANDLING DOUBLE NAMES
        logger.info('Processing duplicate names...')  
        # Some fighters have identical names, and some fighters are 
        # mentioned double in dff. 
        # To distinguish between identically named fighters, we need
        # to process weightclass first, and link weight in dff to 
        # weightclass in dff. We will one-hot encode weight class.
        # Weightclass processing is not finished here, but we will do so later.  

        dfo['WEIGHTCLASS'] = dfo['WEIGHTCLASS'].str.strip() 
        classes = ['Catch ','Straw','Fly','Bantam','Feather','Light','Welter','Middle','Light Heavy'] 
        wclasses = [wclass+'weight' for wclass in classes]+['Heavyweight'] 
        pounds = [0,115,125,135,145,155,170,185,205,265]   

        for wc in classes: 
            dfo[f'{wc}weight'] = (dfo['WEIGHTCLASS'].str.contains(wc)).astype(int) 

        dfo['Heavyweight'] = ( 
            dfo['WEIGHTCLASS'].str.contains('Heavy') & 
            ~dfo['WEIGHTCLASS'].str.contains('Light Heavy')
        ).astype(int)

        # Now let's drop duplicates 
        dff.drop(columns=['URL'],inplace=True)
        # And annoying tony johnson (not anthony) who didnt fight in ufc 


        #dff.drop_duplicates(inplace=True) 
        cols = ['HEIGHT','WEIGHT','REACH','DOB'] 


        dff_d = dff[dff['FIGHTER'].duplicated(keep=False)].copy() 
        dff_d['WEIGHT'] = dff_d['WEIGHT'].str.replace('lbs.','').str.strip()

        for dupe_name in dff_d['FIGHTER'].unique():
            # Get sub df of dupe 
            dff_dn = dff_d[dff_d['FIGHTER'] == dupe_name].copy() 

            # Some random guy that never fought in the UFC and has weight 160
            # and that has the same name as the real Jean Silva. 
            if dupe_name == 'Jean Silva':
                dff.drop(dff_dn[dff_dn['WEIGHT']=='160'].index,inplace=True)
                continue 
            elif dupe_name == 'Victor Valenzuela':
                dff.drop(dff_dn[dff_dn['WEIGHT'] == '155'].index, inplace=True) 
                height = "5'" + ' 9"'
                dff.loc[dff_dn[dff_dn['WEIGHT']!='155'].index, 'HEIGHT'] = height
                continue


            if len(dff_dn) > 2: 
                raise Exception(f'{len(dff_dn)} fighters have name {dupe_name}..')                 

            opt1 = dff_dn[cols].iloc[0].values 
            opt2 = dff_dn[cols].iloc[1].values 
            opt1 = opt1[opt1 != '--'] 
            opt2 = opt2[opt2 != '--'] 
            if all(feat in opt2 for feat in opt1): 
                dff.drop(dff_dn.index[0],inplace=True)  
                continue   
            elif all(feat in opt1 for feat in opt2): 
                opt1 = dff_dn[cols].iloc[0]
                opt2 = dff_dn[cols].iloc[1]
                dff.drop(dff_dn.index[1],inplace=True)
                continue   
            
            # Convert pounds to 'Xweight'
            dff_dn['WEIGHT'] = dff_dn['WEIGHT'].str.replace('--','0').astype(int)
            if 0 in dff_dn['WEIGHT']: 
                raise Exception('Verzin wat leuks') 
            dff_dn['WEIGHT'] = dff_dn['WEIGHT'].apply(lambda wc: wclasses[pounds.index(wc)])
            
            # Open fighters' bouts from dfo... 
            dfo_dn = dfo[dfo['BOUT'].str.contains(dupe_name)][wclasses+['BOUT']]
            if len(dfo_dn) == 0 and not self.new_fights_only:
                # In this case, dupe_name is mentioned twice in in ufc_fighter_tott, 
                # but zero times in ufc_fight_results. This happens either when 
                # dupe_name never fought in the UFC (e.g. Tony Johnson) or when
                # dupe_name fights at the next upcoming bout. 

                if dupe_name != 'Tony Johnson':  
                    logger.warning(f'Data processor has detected a duplicate name (1/4)')
                    logger.warning(f"for fighter(s) {dupe_name} that has either never fought in the UFC (2/4)")
                    logger.warning(f"or that fights at the next upcoming bout (3/4)")
                    logger.warning(f"In case of the latter, please hard code a solution (4/4).")
                    raise Exception(f'See logger') 
                continue 
            dfo_nb = dfo_dn.drop(columns=['BOUT']) 
            dfo_wcs = dfo_nb.columns[(dfo_nb == 1).any()]
            if dfo_wcs.str.contains('Catch').any(): 
                raise Exception('Oh boy you are gonna love this one!') 

            # Okay, fuck it, why make it hard on myself?? 
            if dupe_name == 'Joey Gomez': 
                bad_joey_idx = dff_dn[dff_dn['WEIGHT']=='Lightweight'].index
                dff.drop(bad_joey_idx,inplace=True)
            elif dupe_name == 'Michael McDonald': 
                dff.drop(dff_dn[dff_dn['WEIGHT']!='Bantamweight'].index,inplace=True)  
            elif dupe_name == 'Bruno Silva': 
                b_mid = 'Bruno Middle Silva'
                b_fly = 'Bruno Fly Silva'
                m_idx = dff_dn[dff_dn['WEIGHT']=='Middleweight'].index
                f_idx = dff_dn[dff_dn['WEIGHT']!='Middleweight'].index
                dff.loc[m_idx, 'FIGHTER'] = dff.loc[m_idx, 'FIGHTER'].str.replace(dupe_name,b_mid)
                dff.loc[f_idx, 'FIGHTER'] = dff.loc[f_idx, 'FIGHTER'].str.replace(dupe_name,b_fly) 

                # NB overwriting
                m_idx = dfo_dn[dfo_dn['Middleweight']=='1'].index
                f_idx = dfo_dn[dfo_dn['Middleweight']!='1'].index
                dfo['old BOUT'] = dfo['BOUT'] 
                for collie in ['BOUT', 'name f1', 'name f2']: 
                    dfo.loc[m_idx, collie] = dfo.loc[m_idx, collie].str.replace(dupe_name,b_mid)
                    dfo.loc[f_idx, collie] = dfo.loc[f_idx, collie].str.replace(dupe_name,b_fly)
                dfs_map = dict(zip(dfo['old BOUT'], dfo['BOUT']))
                dfs['BOUT'] = dfs['BOUT'].map(dfs_map).fillna(dfs['BOUT'])    
                dfo.drop(columns=['old BOUT'], inplace=True) 
            else: 
                logger.warning(f'{dupe_name},{dff_dn["WEIGHT"].values}, {dfo_wcs}')
                raise Exception(dupe_name, 'Doe wat leuks with {dupe_name}')  

        # Now make sure that each bout has a unique BOUT value. 
        # Not all bouts have a unique BOUT value. For instance, 
        # when two fighters have a rematch. Fortunately, each
        # bout has the same BOUT and EVENT value in both ufc_fight_stats 
        # and ufc_fight_results so they are easy to match.  

        dfo['BOUT EVENT'] = dfo['BOUT'] + '_' + dfo['EVENT'] 
        dfs['BOUT EVENT'] = dfs['BOUT'] + '_' + dfs['EVENT']
        dfo['BOUT'] = dfo['name f1'] + ' vs. ' + dfo['name f2'] 

        bout_map = dict(zip(dfo['BOUT EVENT'], dfo['BOUT']))
        dfs['BOUT'] = dfs['BOUT EVENT'].map(bout_map)
        del bout_map

        # Now we can give each bout a unique fight id
        # in stead of BOUT EVENT. We can use this to match 
        # dfo and dfs easier and is used for folding later. 
        dfo['BOUT EVENT'] = dfo['BOUT'] + '_' + dfo['EVENT'] 
        dfs['BOUT EVENT'] = dfs['BOUT'] + '_' + dfs['EVENT'] 

        dfo.reset_index(inplace=True)
        dfo.rename(columns={'index': 'f_id'}, inplace=True)

        id_map = dict(zip(dfo['BOUT EVENT'], dfo['f_id']))
        dfs['f_id'] = dfs['BOUT EVENT'].map(id_map)  

        dfs.drop(columns=['BOUT EVENT'], inplace=True) 
        dfo.drop(columns=['BOUT EVENT'], inplace=True)
        del id_map 

        #### PROCESS TEMPORAL DATA
        logger.info('Processing temporal data...') 
        # We now need to match names in dfo (outcomes) to dff (fighters) to access age, height, et... 
        # We already checked if the name is mentioned in dft (alt spellings). 
        # For those that aren't, we now try fuzzywuzzy.
        no_matches = [[],[]]
        for i in [1,2]: 
            problem_names = dfo.loc[~dfo[f'name f{i}'].isin(dff['FIGHTER']),f'name f{i}'].unique()
            choices = dff[~dff['FIGHTER'].isin(dfo[f'name f{i}'])]['FIGHTER'].values 

            matches = [get_best_match(problem_name, choices, threshold=80) for problem_name in problem_names] 
            no_matches[i-1] = [name for name, score in matches if score == 0] 
            if len(no_matches[i-1]) > 0: 
                matches = [match[0] for match in matches]
                dff['FIGHTER'] = dff['FIGHTER'].map(dict(zip(matches,problem_names))).fillna(dff['FIGHTER']) 

        if len(no_matches[0]) > 0 or len(no_matches[1]) > 0:
            # When e.g. women get married, new fights in dfo will get updated 
            # but dff not. Or sometimes "cage names" are used.  
            no_matches = no_matches[0] + no_matches[1]
            no_matches = pd.DataFrame({'dfo name': no_matches,
                                       'dff name': [np.nan]*len(no_matches)})
            dft = pd.concat([dft, no_matches], axis = 0, ignore_index=True) 

            try:
                store_csv(dft, self.interim_data_path / 'alternative_spellings_internal.csv') 
            except Exception as e:
                logger.error(f"Google dff names of {no_matches}")
                raise
            return None

        del choices 

        # Truncate fighters that only fought before UFC 31. 
        # dff = dff[dff['FIGHTER'].isin(pd.concat([dfo['name f1'],dfo['name f2']]))]

        # Convert event date and fighter's DOB to pd.datetime 
        date_format = "%B %d, %Y"
        dfe['DATE'] = dfe['DATE'].str.strip()
        dfe['DATE'] = pd.to_datetime(dfe['DATE'], format=date_format) 
        dfo['DATE'] = dfo['EVENT'].map(dict(zip(dfe['EVENT'], dfe['DATE'])))

        dff_dt_format = "%b %d, %Y"
        dff.loc[dff['DOB']=='--','DOB'] = np.nan 
        dff['DOB'] = pd.to_datetime(dff['DOB'].str.strip(), format=dff_dt_format)

        # Convert age to weeks old 
        for i in [1,2]:
            dfo[f'age f{i}'] = dfo[f'name f{i}'].map(dict(zip(dff['FIGHTER'], dff['DOB'])))
            dfo[f'age f{i}'] = (dfo['DATE'] - dfo[f'age f{i}']).dt.days // 7
            dfo[f'age f{i}'] = dfo[f'age f{i}'].round()

        # Convert event date to weeks since UFC 31.  
        t0 = pd.to_datetime('May 04, 2001', format=date_format)
        dfe['DATE'] = (dfe['DATE']-t0).dt.days // 7
        dfe['DATE'] = dfe['DATE'].round()
        tau_map = dict(zip(dfe['EVENT'],dfe['DATE']))  
        dfo['tau'] = dfo['EVENT'].map(tau_map)
        del tau_map
        dfo.drop(columns=['DATE'], inplace=True)
        
        # Get reach (inches) and height (cms) 
        dff['REACH'] = dff['REACH'].str.strip()
        dff['REACH'] = dff['REACH'].replace('--',np.nan) 
        dff['REACH'] = dff['REACH'].str.replace('"','').astype('Int64')  

        dff['HEIGHT'] = dff['HEIGHT'].str.strip()
        def _process_height(height_str): 
            """
            Calculates height string e.g. str(8' 9") feet to an int in cm.
            """
            if height_str == '--': 
                return np.nan 
            else: 
                height = height_str.replace('"',"").split(' ') 
                height = 30.48*int(height[0][:-1]) + 2.54*int(height[1]) 
                return height 
        dff['HEIGHT'] = dff['HEIGHT'].apply(_process_height) 
        
        for i in [1,2]: 
            dfo = pd.merge(dfo, dff[['FIGHTER','REACH','HEIGHT']], 
                           left_on=f'name f{i}', right_on='FIGHTER', 
                           how='left') 
            dfo[f'reach f{i}'] = dfo['REACH'] 
            dfo[f'height f{i}'] = dfo['HEIGHT'] 
            dfo.drop(columns=['FIGHTER','REACH','HEIGHT'], inplace=True)

        # Getting time format (one-hot encoded) 
        time_format = dfo['TIME FORMAT'].str.split(' ').str[0].astype(int)  
        dfo['3 rounds'] = (time_format == 3).astype(int) 
        dfo['5 rounds'] = (time_format == 5).astype(int) 
        del time_format

        # Convert final round time to seconds
        dfo['TIME'] = dfo['TIME'].str.strip().str.split(':') 
        dfo['TIME'] = 60*dfo['TIME'].str[0].astype(int) + dfo['TIME'].str[1].astype(int) 

        # For each round i a column that says how long round i lasted
        # This will be used for control time and round averages. 
        rounds_fought = dfo['ROUND'].to_numpy()
        final_round_times = dfo['TIME'].to_numpy() 
        rounds_times = np.zeros((len(dfo), 5))
        rounds_range = np.arange(1,6)
        full_rounds_mask = rounds_range[None, :] < rounds_fought[:, None] 
        rounds_times[full_rounds_mask] = 300
        rounds_times[np.arange(len(dfo)), rounds_fought-1] = final_round_times 
        for ri in range(1,6): 
            dfo[f'round {ri} fought'] = (dfo['ROUND'] >= ri).astype(int) 
            dfo[f'r{ri} time'] =  rounds_times[:, ri-1]
            
        # FINISH PROCESSING WEIGHTCLASS, GET SEX, TITLE BOUT
        logger.info('Getting fighter specific info...')  
        # All three will be one-hot encoded.
        # We do this because we will center all features later. 
        # First do weightclass. 

        # Now sex.
        dfo['female'] = (dfo['WEIGHTCLASS'].str.contains('Women')).astype(int)
        dfo['male'] = (~dfo['WEIGHTCLASS'].str.contains('Women')).astype(int)
        
        # Since Catch Weight doesn't mention sex, we need to infer the sex from other (non-Catch weight) bouts.
        # If there are bouts between fighters that only have catch weight bouts, we need to do do it manually.
        # In the future use some AI that can recognize female names. 
        sex_IDd_names = set(dfo[~dfo['WEIGHTCLASS'].str.contains('Catch')]['name f1'].unique()) | \
                        set(dfo[~dfo['WEIGHTCLASS'].str.contains('Catch')]['name f2'].unique())

        is_catch = dfo['WEIGHTCLASS'].str.contains('Catch')
        dfo_catch = dfo[is_catch] 
        catch_only = dfo[(~dfo_catch['name f1'].isin(sex_IDd_names)) & (~dfo['name f2'].isin(sex_IDd_names))]
        if len(catch_only)> 0: 
            catch_only['male'] = [np.nan]*len(catch_only) 
            store_csv(catch_only, self.interim_data_path / 'unknown_sex.csv') 
            try:
                store_csv(dft, self.interim_data_path / 'unknown_sex.txt')
            except Exception as e:
                logger.error(f"Fighters without sex {catch_only}")
                raise
            return None

            raise Exception('Unknown sex {catch_only}')

        # From here, either name f1 or name f2 or both have 1 or more non-Catch weight bouts, so we can use 
        # the sexes these bouts to determine the sex of the fighters for the Catch weight bout.  
        # Get dfo where only name f1 has normal bouts 

        # Get all catch weight bouts
        dfo_catch = dfo[is_catch]

        # Create master fighter-to-sex maps from all normal bouts
        normal_bouts = dfo[~is_catch]

        # Combine f1 and f2 into one series for mapping
        fighters = pd.concat([normal_bouts['name f1'], normal_bouts['name f2']])
        female_values = pd.concat([normal_bouts['female'], normal_bouts['female']])
        male_values = pd.concat([normal_bouts['male'], normal_bouts['male']])

        # Create dictionaries mapping fighter name -> sex
        fighter_female_map = dict(zip(fighters, female_values))
        fighter_male_map = dict(zip(fighters, male_values))

        # Apply mappings to catch weight bouts
        # First try mapping based on name f1
        dfo.loc[dfo_catch.index, 'female'] = dfo_catch['name f1'].map(fighter_female_map)
        dfo.loc[dfo_catch.index, 'male'] = dfo_catch['name f1'].map(fighter_male_map)

        # If name f1 was not in map, try name f2
        dfo.loc[dfo_catch.index, 'female'] = dfo_catch['name f2'].map(fighter_female_map).combine_first(
            dfo.loc[dfo_catch.index, 'female']
        )
        dfo.loc[dfo_catch.index, 'male'] = dfo_catch['name f2'].map(fighter_male_map).combine_first(
            dfo.loc[dfo_catch.index, 'male']
        )

        # Clean up temporary variables
        del normal_bouts, fighters, female_values, male_values, fighter_female_map, fighter_male_map
        """
        dfo_catch_f1 = dfo_catch[dfo_catch['name f1'].isin(sex_IDd_names)]
        dfo_map = dfo[
                ((dfo['name f1'].isin(dfo_catch_f1['name f2'])) | 
                 (dfo['name f2'].isin(dfo_catch_f1)))
                & (~is_catch)]
        map_keys = pd.concat([dfo_map['name f1'], dfo_map['name f2']])

        female_map_values = pd.concat([dfo_map['female'], dfo_map['female']]) 
        female_map = dict(zip(map_keys, female_map_values))
        dfo.loc[dfo_catch_f1.index, 'female'] = dfo.loc[dfo_catch_f1.index, 'female'].map(female_map)

        male_map_values = pd.concat([dfo_map['male'], dfo_map['male']])
        male_map = dict(zip(map_keys, male_map_values))
        dfo.loc[dfo_catch_f1.index, 'male'] = dfo.loc[dfo_catch_f1.index, 'male'].map(male_map)

        if len(dfo_catch_f1) < len(dfo_catch):
            # Only fighter 2 has normal bouts 
            dfo_catch_f2 = dfo_catch[~dfo_catch.index.isin(dfo_catch_f1.index)] 
            dfo_map = dfo[((dfo['name f1'].isin(dfo_catch_f2['name f2'])) |
                           (dfo['name f2'].isin(dfo_catch_f2)))
                           & (~is_catch)]
            map_keys = pd.concat([dfo_map['name f1'], dfo_map['name f2']])

            female_map_values = pd.concat([dfo_map['female'], dfo_map['female']])
            female_map = dict(zip(map_keys, female_map_values))
            dfo.loc[dfo_catch_f2.index, 'female'] = dfo.loc[dfo_catch_f2.index, 'female'].map(female_map)

            male_map_values = pd.concat([dfo_map['male'], dfo_map['male']]) 
            male_map = dict(zip(map_keys, male_map_values))
            dfo.loc[dfo_catch_f2.index, 'male'] = dfo.loc[dfo_catch_f2.index, 'male'].map(male_map)

        del map_keys,female_map_values,male_map_values,female_map,male_map,is_catch,dfo_catch  
        """


        # Now title bout. Interim is counted as Championship bout. 
        dfo['normal bout'] = (~dfo['WEIGHTCLASS'].str.contains('Title')).astype(int) 
        dfo['title bout'] = (dfo['WEIGHTCLASS'].str.contains('Title')).astype(int)

        # FIGHT STATS
        logger.info('Processing fight stats...') 
		# Rename base features 
        raw_cols = ['SIG.STR.','TOTAL STR.','TD','HEAD','BODY','LEG',
                    'DISTANCE','CLINCH','GROUND']
        new_cols = stat_names(which=[''],extras=False,rounds=False,fighters=False) 

        for raw_col, new_col in zip(raw_cols, new_cols): 
            dfs[raw_col] = dfs[raw_col].str.strip().str.split('of')
            dfs[f'{new_col}l'] = dfs[raw_col].str[0].astype(int) # landed  
            dfs[f'{new_col}'] = dfs[raw_col].str[1].astype(int)  # attempted 
            dfs[f'{new_col}A'] = dfs[f'{new_col}l'] / dfs[f'{new_col}'] # accuracy 
        
        new_cols_ext = stat_names(which=['','l','A'], extras=True, rounds=False, fighters=False)

		# Control Time to seconds 
        dfs['CT'] = dfs['CTRL'].str.strip().str.split(':') 
        dfs['CT'] = 60*dfs['CT'].str[0].astype(int) + dfs['CT'].str[1].astype(int)
        dfs['SA'] = dfs['SUB.ATT']
        raw_cols.remove('TD') 
        dfs.drop(columns = raw_cols + ['SUB.ATT','CTRL'], inplace=True)

		# Get all stats to dfo for each fighter, for each round  

        def _stats_to_dfo(f_id, fin_r): 
            """
			Flattens dfs (1 row for for each fighter, for each round) 
			into 1d for dfo (1 row for each bout). It does this for all 
            columns in new_cols_ext and appends np.nan for rounds higher 
            than the final round.  
			
			Args: 
                dfo['f_id']: The bout ID 
                dfo['ROUND']: The number of rounds the fight lasted  

			Returns: 
				All features for both fighters for all rounds (dict) 
			"""
            dfs_id = dfs[dfs['f_id']==f_id].reset_index(drop=True) 
            dfo_stats = {}  
            for fi in range(2): 
                for rj in range(5):
                    if rj < fin_r: 
                        new_stats = dfs_id.loc[fi*fin_r + rj, new_cols_ext].to_numpy()
                    else: 
                        new_stats = [np.nan]*len(new_cols_ext)
                    for new_col, new_stat in zip(new_cols_ext, new_stats): 
                        dfo_stats[f'{new_col}r{rj+1}f{fi+1}'] = new_stat
            return dfo_stats

        stats = dfo.apply(lambda r: _stats_to_dfo(r['f_id'], r['ROUND']), axis=1)
        dfo = pd.concat([dfo, stats.apply(pd.Series)], axis=1)		

        # Get defense accuracy
        A1_cols = stat_names(which=['A'], rounds=True, extras=False, fighters=[1]) 
        D1_cols = stat_names(which=['D'], rounds=True, extras=False, fighters=[1]) 
        A2_cols = stat_names(which=['A'], rounds=True, extras=False, fighters=[2]) 
        D2_cols = stat_names(which=['D'], rounds=True, extras=False, fighters=[2]) 

        dfo[D1_cols] = 1-dfo[A2_cols].values
        dfo[D2_cols] = 1-dfo[A1_cols].values
            
		# Divide stats by round time and compute averages.
        # Note that, say Body Strikes (BS) = 0 for a certain round, 
        # then it should be taken into account for the average attempted and landed strikes, 
        # but not for the (defense) accuracy. 

        cols = np.array(stat_names(which=['','l','A','D'])).reshape(-1,5,2)  # shape: n_stats x 5 rounds x 2 fighters
        r_cols = [f'r{ri} time' for ri in range(1,6)]
        dfo['bout time'] = dfo[r_cols].apply(lambda x: x[r_cols].sum(), 
                                             axis = 1)
        


        new_cols_df = pd.DataFrame(index=dfo.index)  # placeholder for all computed columns
        d_avgs = {}
        # I have to do this differently. 
        # I weigh by how long the round lasted and divide by total bout time, but in some rounds stats 
        # that are not measured are nan, so the weight is off.  

        for sj in range(cols.shape[0]):
            base_stat = cols[sj,0,0].split('r')[0]

            stat_cols_f1 = cols[sj, :, 0].tolist()
            stat_cols_f2 = cols[sj, :, 1].tolist()

            wstats_f1 = dfo[stat_cols_f1].values * dfo[r_cols].values
            wstats_f2 = dfo[stat_cols_f2].values * dfo[r_cols].values

            d_avgs[f'{base_stat}avgf1'] = pd.DataFrame(wstats_f1, index=dfo.index).sum(axis=1, skipna=True, min_count=1) / dfo['bout time'] 
            d_avgs[f'{base_stat}avgf2'] = pd.DataFrame(wstats_f2, index=dfo.index).sum(axis=1, skipna=True, min_count=1) / dfo['bout time'] 

        dfo = pd.concat([dfo, pd.DataFrame(d_avgs)], axis=1)

        # We store the cleaned ufc_fighter_tott.csv because we need it for 
        # processing fights from upcoming events where at least one fighter
        # is debuting. 
        if not self.new_fights_only:
            names = pd.concat([dfo['name f1'], dfo['name f2']]).reset_index(drop=True)
            ages  = pd.concat([dfo['age f1'], dfo['age f2']]).reset_index(drop=True)
            dff['age'] = dff['FIGHTER'].map(dict(zip(names, ages))).fillna(np.nan)
            store_csv(dff, self.interim_data_path / 'ufc_fighter_tott_clean.csv') 


        # OUTCOME AND METHOD
        logger.info('Creating classes...') 

		# dfo['OUTCOME'] in [W/L, L/W, D/D, NC/NC] 
        if dfo['OUTCOME'].isna().any(): 
            raise Exception(f'OUTCOME contains Nans. {dfo["OUTCOME"].isna().index}') 
        if dfo['METHOD'].isna().any(): 
            raise Exception(f'OUTCOME contains Nans. {dfo["METHOD"].isna().index}') 

        dfo['outcome'] = dfo['OUTCOME'].str.split('/').str[0]
        dfo['result'] = dfo['METHOD'].str.strip()
        dfo.drop(columns=['METHOD','OUTCOME'],inplace=True) 
        ko_types = ['KO/TKO', "TKO - Doctor's Stoppage"]
        dfo.loc[(dfo['outcome'] == 'W') & 
                (dfo['result'].isin(ko_types)), 'result'] = 0 
        dfo.loc[(dfo['outcome'] == 'L') & 
                (dfo['result'].isin(ko_types)), 'result'] = 6 
        dfo.loc[(dfo['outcome'] == 'W') & 
                (dfo['result']=='Submission'), 'result'] = 1
        dfo.loc[(dfo['outcome'] == 'L') & 
                (dfo['result']=='Submission'), 'result'] = 5
        dfo.loc[(dfo['outcome'] == 'W') & 
                (dfo['result'].str.contains('Decision')), 'result'] = 2
        dfo.loc[(dfo['outcome'] == 'L') & 
                (dfo['result'].str.contains('Decision')), 'result'] = 4	
        dfo.loc[dfo['outcome'] == 'D', 'result'] = 3 
        dfo.loc[dfo['outcome'] == 'NC', 'result'] = -1 
        dfo.loc[dfo['result'] == 'DQ', 'result'] = -1 

        logger.info("FINITOOOOOOOOOOO") 
        print(type(dfo))
        return dfo 

def process_all_data(prefer_external=True, new_fights_only=False) -> None:
    """
    Main function to process all UFC data.
    
    This function orchestrates the entire data processing pipeline:
    1. Load raw data
    2. Clean individual datasets  
    3. Merge datasets 
    4. Create basic features  
    5. Save processed data
    """
    logger.info("Starting UFC data processing pipeline")
    
    processor = UFCDataProcessor(prefer_external, new_fights_only) 
    
    try:
        logger.info("Loading raw data")
        processor.load_raw_data() 
        
        logger.info('Starting data cleaning.') 
        clean_data = processor.clean_data() 

        logger.info('Storing raw data') 
        clean_data_path = processor.interim_data_path / "clean_ufcstats-com_data.csv"
        if new_fights_only:
            clean_data_old = open_csv(clean_data_path)
            clean_data = pd.concat([clean_data, clean_data_old])
            clean_data = clean_data.drop_duplicates() 
            clean_data.reset_index(drop=True, inplace=True)
        clean_data.sort_values(by=['tau','name f1'], inplace=True)
        clean_data.reset_index(drop=True) 
        store_csv(clean_data, processor.interim_data_path / "clean_ufcstats-com_data.csv")
        
        logger.info("UFC data processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        raise


if __name__ == "__main__":
    process_all_data() 
