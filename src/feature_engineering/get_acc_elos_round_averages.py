"""
Wrapper to generate accuracy ELOs for round averages.
FeatureManager can import `get_acc_elos_round_averages` to produce
`acc_elos_round_averages.csv` using non-per-round K values.
"""

from src.feature_engineering.get_acc_elos import get_acc_elos


def get_acc_elos_round_averages(which_K: str = 'log',
                                process_upcoming_fights: bool = False):
    return get_acc_elos(per_round=False,
                        which_K=which_K,
                        process_upcoming_fights=process_upcoming_fights)


if __name__ == '__main__':
    get_acc_elos_round_averages()
