"""
Wrapper to generate per-round stat ELOs.
This exists so FeatureManager can import `get_stat_elos_per_round` to produce
`stat_elos_per_round.csv` using per-round K-values.
"""

from src.feature_engineering.get_stat_elos import get_stat_elos


def get_stat_elos_per_round(exact_score: bool = True,
                            process_upcoming_fights: bool = False,
                            which_K: str = 'log',
                            always_update: bool = False):
    return get_stat_elos(per_round=True,
                         exact_score=exact_score,
                         process_upcoming_fights=process_upcoming_fights,
                         which_K=which_K,
                         always_update=always_update)


if __name__ == '__main__':
    get_stat_elos_per_round()
