from src.feature_engineering.get_stat_elos import get_stat_elos 

def get_stat_elos_round_averages(exact_score=True, 
                            process_upcoming_fights=False,
                            always_update=False,
                            which_K = 'log'): 
               
    get_stat_elos(per_round=False,
                  exact_score=exact_score,
                  always_update=always_update, 
                  process_upcoming_fights=process_upcoming_fights,
                  which_K = which_K
    ) 


