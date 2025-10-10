import sys
import traceback
from pathlib import Path

# Ensure workspace root is on sys.path so `src` can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_selection.trainvalpred import TrainValPred
from src.utils.general import get_data_path, open_csv


def main():
    print("[smoke] starting symmetrize(pred_only=True)...")
    tvp = TrainValPred(feature_sets={})
    # Expect processed train/valid/pred to exist in repo
    proc = get_data_path('processed')
    train_p = proc / 'train.csv'
    valid_p = proc / 'valid.csv'
    pred_p = proc / 'pred.csv'

    for p in (train_p, valid_p, pred_p):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # Run the path that shouldn't alter train/valid here; only writes pred_symm.csv
    tvp.symmetrize(for_svd=False, pred_only=True)

    out_p = proc / 'pred_symm.csv'
    if not out_p.exists():
        raise FileNotFoundError(f"Expected output not found: {out_p}")

    df = open_csv(out_p)
    print(f"[smoke] pred_symm.csv -> rows={len(df)}, cols={df.shape[1]}")
    print("[smoke] SUCCESS")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("[smoke] FAILED:", e)
        traceback.print_exc()
        sys.exit(1)
