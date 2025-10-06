"""
Rock-Paper-Scissors style feature engineering on opponent overlap.

Goal:
- For each fight, look at each fighter's past opponents in recent time windows.
- Split by result type (KO/Sub/Dec) and W/L, and also add overall aggregates:
    - all previous opponents fought (regardless of result)
    - all previous opponents beaten (wins-any)
    - all previous opponents lost to (losses-any)
- Count overlap (set intersection) between fighter1 and fighter2 lists for each bucket.

Notes:
- This rewrite uses CleanedFights.get_record() from src/data_processing/cleaned_data.py
    to gather past opponents/results efficiently, then slices by years-ago windows.
- First implements process_upcoming_fights=False path.
"""

from typing import List, Tuple, Dict as TDict

import pandas as pd
from loguru import logger

from src.utils.general import get_data_path, store_csv
from src.data_processing.cleaned_data import CleanedFights
from src.data_processing.upcoming_fights import UpcomingFights 

class RockPaperScissor:
    """
    Builds opponent overlap features between fighter 1 and fighter 2.

    intervals: list of year-boundaries, e.g. [0, 2, 5] meaning (0,2], (2,5] years-ago windows.
    process_upcoming_fights: if True, will be implemented after training path is validated.
    """

    def __init__(self, intervals: List[float], process_upcoming_fights: bool = False):
        self.intervals = intervals
        self.process_upcoming_fights = process_upcoming_fights

    @staticmethod
    def _slice_lists_by_years(cur_tau: int, past_taus: List[int], past_opps: List[str], past_res: List[int],
                              t1: float, t2: float) -> Tuple[List[str], List[int]]:
        if past_taus is None or len(past_taus) == 0:
            return [], []
        years_ago = [max(0.0, (cur_tau - pt) / 52.0) for pt in past_taus]
        opps, res = [], []
        for i, ya in enumerate(years_ago):
            if t1 < ya <= t2:
                opps.append(past_opps[i])
                res.append(past_res[i])
        return opps, res

    @staticmethod
    def _build_lists_for_interval(opps: List[str], res: List[int], ival: str) -> TDict[str, List[str]]:
        # Result codes: 0 KOw, 1 Subw, 2 Decw, 3 Draw, 4 Decl, 5 Subl, 6 KOl
        buckets: TDict[str, List[str]] = {}

        def put(key: str, vals: List[str]):
            buckets[key] = vals if vals is not None else []

        # Per-method w/l
        put(f"KOw_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 0])
        put(f"KOl_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 6])
        put(f"Subw_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 1])
        put(f"Subl_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 5])
        put(f"Decw_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 2])
        put(f"Decl_{ival}_f1", [o for i, o in enumerate(opps) if res[i] == 4])

        # Aggregates requested by user
        put(f"wins_any_{ival}_f1", [o for i, o in enumerate(opps) if res[i] in (0, 1, 2)])
        put(f"losses_any_{ival}_f1", [o for i, o in enumerate(opps) if res[i] in (4, 5, 6)])
        put(f"fought_any_{ival}_f1", list(opps))  # irrespective of result

        return buckets

    @staticmethod
    def _count_overlap(a: List[str], b: List[str]) -> int:
        if not a or not b:
            return 0
        # set-intersection (ignore multiple bouts vs same opponent)
        return int(len(set(a) & set(b)))

    def construct(self) -> pd.DataFrame:
        if self.process_upcoming_fights:
            # Minimal upcoming fights support: align with base_features pattern
            # We load historical cleaned fights and upcoming matchups to compute overlaps up to current time.
            logger.info("Loading historical cleaned fights and upcoming fights...")
            dfo_hist = CleanedFights()
            dfo_hist = dfo_hist[~dfo_hist['result'].isin([-1, 3])].reset_index(drop=True)

            dfp = UpcomingFights().open_clean()
            # Identify fighters with/without history
            known_names = set(pd.concat([dfo_hist['name f1'], dfo_hist['name f2']], ignore_index=True).astype(str).str.strip().unique().tolist())
            dfp = dfp.copy()
            dfp['__has_hist_f1'] = dfp['name f1'].astype(str).str.strip().isin(known_names)
            dfp['__has_hist_f2'] = dfp['name f2'].astype(str).str.strip().isin(known_names)

            n_deb_f1 = int((~dfp['__has_hist_f1']).sum())
            n_deb_f2 = int((~dfp['__has_hist_f2']).sum())
            if n_deb_f1 + n_deb_f2 > 0:
                logger.warning(f"Upcoming RPS: debutants detected - f1 without history: {n_deb_f1}, f2 without history: {n_deb_f2}")

            # We will compute RPS features for dfp by reusing the same pipeline but restricting to past fights only
            # Prepare a combined frame with same columns, where upcoming rows appear with current tau but no past lists
            dfo_hist = dfo_hist.reset_index(drop=True)
            dfo_hist['row_id'] = dfo_hist.index
            dfo_hist['tau_cur'] = dfo_hist['tau']

            # Compute fighter1 lists from history
            dfr_a = CleanedFights(dfo_hist).get_record(cols=['name f2', 'result', 'tau'], include_tau=False)
            dfr_a = dfr_a.rename(columns={'name f2': 'past_opps', 'result': 'past_results', 'tau': 'past_taus'})

            # Attach upcoming rows metadata and compute lists for them using the historical mapping (empty lists ok)
            dfp['row_id'] = range(len(dfp))
            dfp['tau_cur'] = dfp['tau']

            # Build lists per interval for fighter 1 for upcoming fights by mapping on fighter names into historical lists
            ival_labels: List[str] = []
            for j in range(len(self.intervals) - 1):
                t1, t2 = self.intervals[j], self.intervals[j + 1]
                ival = f"{t1}-{t2}ya"
                ival_labels.append(ival)

                def build_f1_for_row(row):
                    # Debutant: no history => empty lists
                    if not bool(row['__has_hist_f1']):
                        return self._build_lists_for_interval([], [], ival)
                    # Find matching historical row indices for the fighter to grab past lists
                    hist_rows = dfr_a[dfr_a['name f1'] == row['name f1']]
                    if len(hist_rows) == 0:
                        return self._build_lists_for_interval([], [], ival)
                    # Use the last available history row to represent past lists
                    hr = hist_rows.iloc[-1]
                    opps, res = self._slice_lists_by_years(int(row['tau_cur']), hr.get('past_taus', []), hr.get('past_opps', []), hr.get('past_results', []), t1, t2)
                    return self._build_lists_for_interval(opps, res, ival)

                lists_df_a = pd.DataFrame(dfp.apply(build_f1_for_row, axis=1).tolist())
                dfp = pd.concat([dfp, lists_df_a], axis=1)

            # Build fighter2 buckets similarly using swapped perspective
            dfo_hist_b = dfo_hist.copy()
            dfo_hist_b[['name f1', 'name f2']] = dfo_hist_b[['name f2', 'name f1']]
            if 'result' in dfo_hist_b.columns:
                dfo_hist_b['result'] = dfo_hist_b['result'].apply(lambda x: 6 - x if pd.notna(x) else x).replace(7, -1)
            dfr_b = CleanedFights(dfo_hist_b).get_record(cols=['name f2', 'result', 'tau'], include_tau=False)
            dfr_b = dfr_b.rename(columns={'name f2': 'past_opps', 'result': 'past_results', 'tau': 'past_taus'})

            for j in range(len(self.intervals) - 1):
                t1, t2 = self.intervals[j], self.intervals[j + 1]
                ival = f"{t1}-{t2}ya"

                def build_f2_for_row(row):
                    if not bool(row['__has_hist_f2']):
                        return self._build_lists_for_interval([], [], ival)
                    hist_rows = dfr_b[dfr_b['name f1'] == row['name f2']]
                    if len(hist_rows) == 0:
                        return self._build_lists_for_interval([], [], ival)
                    hr = hist_rows.iloc[-1]
                    opps, res = self._slice_lists_by_years(int(row['tau_cur']), hr.get('past_taus', []), hr.get('past_opps', []), hr.get('past_results', []), t1, t2)
                    return self._build_lists_for_interval(opps, res, ival)

                lists_df_b = pd.DataFrame(dfp.apply(build_f2_for_row, axis=1).tolist())
                lists_df_b = lists_df_b.rename(columns={c: c.replace('_f1', '_f2') for c in lists_df_b.columns})
                dfp = pd.concat([dfp, lists_df_b], axis=1)

            # Compute overlaps for upcoming fights
            logger.info("Computing overlap counts for upcoming fights...")
            overlap_cols: TDict[str, List[int]] = {}
            total_cols: TDict[str, List[int]] = {}
            categories = [
                'KOw', 'KOl', 'Subw', 'Subl', 'Decw', 'Decl',
                # add aggregate buckets
                'wins_any', 'losses_any', 'fought_any',
            ]
            base_types = ['KOw', 'KOl', 'Subw', 'Subl', 'Decw', 'Decl']

            for ival in ival_labels:
                for cat in categories:
                    c1 = f"{cat}_{ival}_f1"
                    c2 = f"{cat}_{ival}_f2"
                    if c1 not in dfp.columns or c2 not in dfp.columns:
                        continue
                    overlap_cols[f"overlap_{cat}_{ival}"] = dfp.apply(lambda r: self._count_overlap(r[c1], r[c2]), axis=1).astype(int).tolist()
                    total_cols[f"tot_{cat}_{ival}_f1"] = dfp[c1].apply(lambda x: int(len(x) if isinstance(x, list) else 0)).tolist()
                    total_cols[f"tot_{cat}_{ival}_f2"] = dfp[c2].apply(lambda x: int(len(x) if isinstance(x, list) else 0)).tolist()

                for t1 in base_types:
                    c1 = f"{t1}_{ival}_f1"
                    if c1 not in dfp.columns:
                        continue
                    for t2 in base_types:
                        if t2 == t1:
                            continue
                        c2 = f"{t2}_{ival}_f2"
                        if c2 not in dfp.columns:
                            continue
                        feat_name = f"overlap_{t1}_vs_{t2}_{ival}"
                        overlap_cols[feat_name] = dfp.apply(lambda r: self._count_overlap(r[c1], r[c2]), axis=1).astype(int).tolist()

            keep_cols = ['tau', 'name f1', 'name f2']
            out_df = dfp[keep_cols].copy()
            for k, v in overlap_cols.items():
                out_df[k] = pd.Series(v, index=out_df.index, dtype=int)
            for k, v in total_cols.items():
                out_df[k] = pd.Series(v, index=out_df.index, dtype=int)

            eng_cols = [c for c in out_df.columns if c.startswith('overlap_') or c.startswith('tot_')]
            for c in eng_cols:
                out_df[c] = out_df[c].fillna(0).astype(int)

            return out_df

        logger.info("Loading cleaned fights and preparing past opponents/results lists...")
        dfo = CleanedFights()

        # Keep scalar current metadata in separate columns; get_record will overwrite cols with lists
        dfo = dfo.reset_index(drop=True)
        dfo['row_id'] = dfo.index
        dfo['tau_cur'] = dfo['tau']

        # Build lists of past opponents and results for fighter 1 perspective
        dfr_a = CleanedFights(dfo).get_record(cols=['name f2', 'result', 'tau'], include_tau=False)
        dfr_a = dfr_a.rename(columns={'name f2': 'past_opps', 'result': 'past_results', 'tau': 'past_taus'})

        # For each interval in years, create lists per bucket for fighter 1
        ival_labels: List[str] = []
        for j in range(len(self.intervals) - 1):
            t1, t2 = self.intervals[j], self.intervals[j + 1]
            ival = f"{t1}-{t2}ya"
            ival_labels.append(ival)

            def build_for_row_a(row):
                opps, res = self._slice_lists_by_years(int(row['tau_cur']), row['past_taus'], row['past_opps'], row['past_results'], t1, t2)
                return self._build_lists_for_interval(opps, res, ival)

            lists_df_a = pd.DataFrame(dfr_a.apply(build_for_row_a, axis=1).tolist())
            dfr_a = pd.concat([dfr_a, lists_df_a], axis=1)

        # Build lists for fighter 2 perspective by swapping fighters and mirroring result
        dfo_b = dfo.copy()
        dfo_b[['name f1', 'name f2']] = dfo_b[['name f2', 'name f1']]
        if 'result' in dfo_b.columns:
            dfo_b['result'] = dfo_b['result'].apply(lambda x: 6 - x if pd.notna(x) else x).replace(7, -1)

        dfr_b = CleanedFights(dfo_b).get_record(cols=['name f2', 'result', 'tau'], include_tau=False)
        dfr_b = dfr_b.rename(columns={'name f2': 'past_opps', 'result': 'past_results', 'tau': 'past_taus'})

        for j in range(len(self.intervals) - 1):
            t1, t2 = self.intervals[j], self.intervals[j + 1]
            ival = f"{t1}-{t2}ya"

            def build_for_row_b(row):
                opps, res = self._slice_lists_by_years(int(row['tau_cur']), row['past_taus'], row['past_opps'], row['past_results'], t1, t2)
                return self._build_lists_for_interval(opps, res, ival)

            lists_df_b = pd.DataFrame(dfr_b.apply(build_for_row_b, axis=1).tolist())
            # rename f1 suffix to f2 for fighter 2 perspective
            lists_df_b = lists_df_b.rename(columns={c: c.replace('_f1', '_f2') for c in lists_df_b.columns})
            dfr_b = pd.concat([dfr_b, lists_df_b], axis=1)

        # Merge fighter 1 and fighter 2 lists on row_id
        keep_meta = ['tau', 'name f1', 'name f2', 'result', 'row_id']
        dfr = pd.concat([dfo[keep_meta], dfr_a.drop(columns=[col for col in dfr_a.columns if col in keep_meta and col != 'row_id'])], axis=1)
        # Add _f2 list columns from dfr_b (aligning by index)
        f2_list_cols = [c for c in dfr_b.columns if c.endswith('_f2')]
        dfr = pd.concat([dfr, dfr_b[f2_list_cols].reset_index(drop=True)], axis=1)

        # Now compute overlap counts and totals per interval/category
        logger.info("Computing overlap counts and totals...")
        overlap_cols: TDict[str, List[int]] = {}
        total_cols: TDict[str, List[int]] = {}

        def add_series(name: str, values: List[int]):
            overlap_cols[name] = values

        def add_tot(name: str, values: List[int]):
            total_cols[name] = values

        categories = [
            'KOw', 'KOl', 'Subw', 'Subl', 'Decw', 'Decl',
            # add aggregate buckets
            'wins_any', 'losses_any', 'fought_any',
        ]
        # Use the same list for cross-type loops
        base_types = ['KOw', 'KOl', 'Subw', 'Subl', 'Decw', 'Decl']

        for ival in ival_labels:
            for cat in categories:
                c1 = f"{cat}_{ival}_f1"
                c2 = f"{cat}_{ival}_f2"
                if c1 not in dfr.columns or c2 not in dfr.columns:
                    continue
                add_series(f"overlap_{cat}_{ival}", dfr.apply(lambda r: self._count_overlap(r[c1], r[c2]), axis=1).astype(int).tolist())
                add_tot(f"tot_{cat}_{ival}_f1", dfr[c1].apply(lambda x: int(len(x) if isinstance(x, list) else 0)).tolist())
                add_tot(f"tot_{cat}_{ival}_f2", dfr[c2].apply(lambda x: int(len(x) if isinstance(x, list) else 0)).tolist())

            # Add cross-type overlaps, skipping same-type to avoid duplicates
            for t1 in base_types:
                c1 = f"{t1}_{ival}_f1"
                if c1 not in dfr.columns:
                    continue
                for t2 in base_types:
                    if t2 == t1:
                        continue  # skip redundant same-type overlap (equals overlap_{t1}_{ival})
                    c2 = f"{t2}_{ival}_f2"
                    if c2 not in dfr.columns:
                        continue
                    feat_name = f"overlap_{t1}_vs_{t2}_{ival}"
                    add_series(feat_name, dfr.apply(lambda r: self._count_overlap(r[c1], r[c2]), axis=1).astype(int).tolist())

        # Assemble final output: metadata + engineered integer features
        keep_cols = ['tau', 'name f1', 'name f2', 'result']
        out_df = dfr[keep_cols].copy()
        for k, v in overlap_cols.items():
            out_df[k] = pd.Series(v, index=out_df.index, dtype=int)
        for k, v in total_cols.items():
            out_df[k] = pd.Series(v, index=out_df.index, dtype=int)

        # Ensure integer dtype for engineered features
        eng_cols = [c for c in out_df.columns if c.startswith('overlap_') or c.startswith('tot_')]
        for c in eng_cols:
            out_df[c] = out_df[c].fillna(0).astype(int)

        # Done
        return out_df


def get_rock_paper_scissor(intervals: List[float], process_upcoming_fights: bool = False) -> pd.DataFrame:
    rps = RockPaperScissor(intervals, process_upcoming_fights)
    df = rps.construct()

    # Save output
    if process_upcoming_fights:
        UpcomingFights().append_features(df)
        return 0 
        #out_path = get_data_path('features') / 'pred_rock_paper_scissor.csv'
    else:
        out_path = get_data_path('features') / 'rock_paper_scissor.csv'
    store_csv(df, out_path)
    logger.info(f"Saved rock-paper-scissor features to {out_path}")
    return df


if __name__ == '__main__':
    # Example: last 0-2 years window
    intervals = [0, 4]
    get_rock_paper_scissor(intervals, process_upcoming_fights=False)






















