"""
Wrapper to generate accuracy ELOs per round.
FeatureManager can import `get_acc_elos_per_round` to produce
`acc_elos_per_round.csv` using per-round K-values.
"""

from src.feature_engineering.get_acc_elos import get_acc_elos


def get_acc_elos_per_round(which_K: str = 'log',
                           process_upcoming_fights: bool = False):
    return get_acc_elos(per_round=True,
                        which_K=which_K,
                        process_upcoming_fights=process_upcoming_fights)


if __name__ == '__main__':
    get_acc_elos_per_round()
