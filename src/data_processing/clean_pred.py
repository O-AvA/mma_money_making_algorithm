"""
Process upcoming fights from raw/pred_raw.csv into interim/pred_clean.csv.

Rules implemented per request:
- Normalize/strip inputs
- Map names via existing cleaned data names, alt-spellings mapping, then fuzzy match; else keep as-is
- Pull height, reach from last known values in cleaned data
- For new fighters, try to infer height, reach and age (from DOB) using interim/ufc_fighter_tott_clean.csv; warn if still not found
- Infer sex per bout like clean_raw_data.py: from WEIGHTCLASS, infer catch-weight using prior bouts
- Compute tau from DATE (weeks since May 04, 2001)
- Weightclass one-hots, title/normal bout per clean_raw_data.py logic
- Round logic: title bout => 5 rounds; first fight => 5 rounds; others => 3 rounds

Output saved to data/interim/pred_clean.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from loguru import logger
import re

from src.utils.general import (
    get_data_path,
    open_csv,
    store_csv,
    creaopen_file,
    get_best_match,
)


DATE_FMT_LONG = "%B %d, %Y"  # e.g., May 04, 2001
DATE_FMT_SHORT = "%b %d, %Y"  # e.g., May 4, 2001
T0 = datetime.strptime("May 04, 2001", DATE_FMT_LONG)


def _parse_date(date_str: str) -> datetime:
    s = str(date_str).strip()
    for fmt in (DATE_FMT_LONG, DATE_FMT_SHORT, "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    raise ValueError(f"Unrecognized date format: {date_str}")


def _compute_tau(date_str: str) -> int:
    dt = _parse_date(date_str)
    return int(((dt - T0).days) // 7)


def _load_cleaned() -> pd.DataFrame:
    path = get_data_path("interim") / "clean_ufcstats-com_data.csv"
    dfc = open_csv(path)
    if dfc.empty:
        raise FileNotFoundError("clean_ufcstats-com_data.csv not found or empty")
    # Only keep the columns we may need to reference quickly
    needed = [
        "tau",
        "name f1",
        "name f2",
        "age f1",
        "age f2",
        "height f1",
        "height f2",
        "reach f1",
        "reach f2",
        "female",
        "male",
        "WEIGHTCLASS",
    ]
    cols = [c for c in needed if c in dfc.columns]
    dfc = dfc[cols + [c for c in dfc.columns if c not in cols]]  # retain all but ensure needed present
    return dfc


def _load_alt_spellings() -> Dict[str, str]:
    # Combine internal and external alt spellings if available
    dft_int = creaopen_file(get_data_path("interim") / "alternative_spellings_internal.csv")
    dft_ext = creaopen_file(get_data_path("interim") / "alternative_spellings_external.csv")

    def to_map(df: pd.DataFrame) -> Dict[str, str]:
        if df is None or df.empty:
            return {}
        cols = df.columns.str.lower().tolist()
        # Expect columns like 'dfo name', 'dff name'
        try:
            dff_col = next(c for c in df.columns if c.lower().strip() == "dff name")
            dfo_col = next(c for c in df.columns if c.lower().strip() == "dfo name")
            m = dict(zip(df[dff_col].astype(str).str.strip(), df[dfo_col].astype(str).str.strip()))
            # normalize keys to lowercase for matching
            return {k.lower(): v for k, v in m.items() if isinstance(k, str) and isinstance(v, str)}
        except StopIteration:
            return {}

    m_int = to_map(dft_int)
    m_ext = to_map(dft_ext)
    # External overrides internal by default (could adjust priority if desired)
    out = {**m_int, **m_ext}
    return out


# New helpers: load and query interim/ufc_fighter_tott_clean.csv for new fighters
def _load_fighter_totals() -> pd.DataFrame:
    path = get_data_path("interim") / "ufc_fighter_tott_clean.csv"
    df = creaopen_file(path)
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()
    return df


def _build_totals_index(dft: pd.DataFrame) -> pd.DataFrame:
    if dft.empty:
        return dft
    dft = dft.copy()
    # pick a name column
    name_col = next(
        (c for c in dft.columns if str(c).strip().lower() in ("name", "fighter", "fighter name", "fighter_name")),
        None,
    )
    if name_col is None:
        name_col = next((c for c in dft.columns if "name" in str(c).strip().lower()), None)
    if name_col is None:
        dft["__name_norm__"] = ""
        return dft
    dft["__name_norm__"] = dft[name_col].astype(str).str.strip().str.lower()
    return dft


def _lookup_totals_row(dft_idx: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if dft_idx.empty:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    hits = dft_idx.loc[dft_idx["__name_norm__"] == key]
    if hits.empty:
        return None
    # in case of duplicates, take last
    return hits.iloc[-1]


def _to_inches(val: object) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if not s or s in ("nan", "none", "null"):
        return np.nan
    # plain number
    try:
        return float(s)
    except Exception:
        pass
    nums = re.findall(r"(\d+\.?\d*)", s)
    if "cm" in s and nums:
        return float(nums[0]) / 2.54
    if "'" in s and nums:
        # feet'inches"
        feet = float(nums[0])
        inches = float(nums[1]) if len(nums) > 1 else 0.0
        return feet * 12.0 + inches
    if "m" in s and "cm" not in s and nums:
        # meters to inches
        return float(nums[0]) * 39.3701
    if nums:
        return float(nums[0])
    return np.nan


def _extract_totals_metrics(row: pd.Series) -> Tuple[float, float, Optional[datetime]]:
    # find candidate columns by substring
    def find_col(subs: Tuple[str, ...]) -> Optional[str]:
        for c in row.index:
            lc = str(c).strip().lower()
            if any(sub in lc for sub in subs):
                return c
        return None

    h_col = find_col(("height", "ht"))
    r_col = find_col(("reach",))
    d_col = find_col(("dob", "date of birth", "birth date", "born"))

    h = _to_inches(row[h_col]) if h_col else np.nan
    r = _to_inches(row[r_col]) if r_col else np.nan

    dob = None
    if d_col:
        val = row[d_col]
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            try:
                ts = _parse_date(str(val))
            except Exception:
                ts = None
        dob = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
    return h, r, dob


def _age_from_dob_at_tau(dob: datetime, tau_pred: int) -> Optional[int]:
    if dob is None or pd.isna(tau_pred):
        return None
    fight_dt = T0 + timedelta(weeks=int(tau_pred))
    years = (fight_dt - dob).days / 365.25
    return int(np.floor(years))


class NameMapper:
    def __init__(self, dfc: pd.DataFrame, alt_map: Dict[str, str]):
        # Known canonical names (use casing from dfc)
        names1 = dfc["name f1"].dropna().astype(str).str.strip()
        names2 = dfc["name f2"].dropna().astype(str).str.strip()
        known = pd.Index(names1.tolist() + names2.tolist()).unique()
        self.known_names = set(known)
        self.known_lc = {n.lower(): n for n in self.known_names}
        self.alt_map = alt_map or {}

    def map_name(self, raw_name: str) -> Tuple[str, bool]:
        """Return (canonical_name, is_new_fighter).
        Order: exact in dfc -> alt map -> fuzzy -> fallback (new)
        """
        name = str(raw_name).strip()
        if not name:
            return name, True

        # Exact (case-insensitive) in dfc
        lc = name.lower()
        if lc in self.known_lc:
            return self.known_lc[lc], False

        # Alt mapping (keys are lowercased)
        if lc in self.alt_map:
            mapped = self.alt_map[lc]
            # If mapped exists in known names, use canonical case if possible
            mlc = mapped.lower()
            if mlc in self.known_lc:
                return self.known_lc[mlc], False
            return mapped, False

        # Fuzzy match
        match, score = get_best_match(name, list(self.known_names))
        if score > 0:
            return match, False

        # New fighter
        return name, True


def _last_known_value_for_fighter(dfc: pd.DataFrame, name: str, base: str) -> Optional[float]:
    """Get last known value for base in {height, reach, age} for fighter name.
    base columns are like '{base} f1' / '{base} f2'. Return latest non-null by tau.
    If base == 'age', we may adjust later for prediction tau separately.
    """
    name = str(name).strip()
    if not name or ("name f1" not in dfc or "name f2" not in dfc):
        return np.nan
    cols = [f"{base} f1", f"{base} f2"]
    cols = [c for c in cols if c in dfc.columns]
    if not cols:
        return np.nan

    # Select rows where the fighter appears as f1 or f2
    sel_f1 = dfc["name f1"].astype(str).str.strip() == name
    sel_f2 = dfc["name f2"].astype(str).str.strip() == name

    df1 = dfc.loc[sel_f1, ["tau"] + [c for c in cols if c.endswith(" f1")]].rename(columns={f"{base} f1": base})
    df2 = dfc.loc[sel_f2, ["tau"] + [c for c in cols if c.endswith(" f2")]].rename(columns={f"{base} f2": base})

    df = pd.concat([df1[["tau", base]] if base in df1 else pd.DataFrame(),
                    df2[["tau", base]] if base in df2 else pd.DataFrame()],
                   axis=0, ignore_index=True)
    if df.empty:
        return np.nan
    df = df.dropna(subset=[base])
    if df.empty:
        return np.nan
    # Latest by tau
    df = df.sort_values(by="tau")
    return df.iloc[-1][base]


def _last_known_age_and_tau(dfc: pd.DataFrame, name: str) -> Tuple[Optional[float], Optional[int]]:
    name = str(name).strip()
    if not name:
        return np.nan, np.nan
    # Gather ages with their taus
    sel_f1 = dfc["name f1"].astype(str).str.strip() == name
    sel_f2 = dfc["name f2"].astype(str).str.strip() == name
    out = []
    if "age f1" in dfc.columns:
        out.append(dfc.loc[sel_f1, ["tau", "age f1"]].rename(columns={"age f1": "age"}))
    if "age f2" in dfc.columns:
        out.append(dfc.loc[sel_f2, ["tau", "age f2"]].rename(columns={"age f2": "age"}))
    if not out:
        return np.nan, np.nan
    ages = pd.concat(out, axis=0, ignore_index=True)
    ages = ages.dropna(subset=["age"]) if not ages.empty else ages
    if ages.empty:
        return np.nan, np.nan
    ages = ages.sort_values(by="tau")
    row = ages.iloc[-1]
    return float(row["age"]), int(row["tau"]) if not pd.isna(row["tau"]) else np.nan


def _infer_sex_for_bout(dfc: pd.DataFrame, n1: str, n2: str, weightclass: str) -> Tuple[int, int]:
    """Return (male, female) for the bout.
    Logic mirrors clean_raw_data: 'Women' => female; else male.
    For Catch weight, infer from prior non-catch bouts if possible.
    If both fighters debut and catch weight => warn, set male=1,female=0.
    """
    wc = str(weightclass).strip()
    if "Women" in wc:
        return 0, 1

    # Catch: try infer from prior non-catch bouts
    is_catch = "Catch" in wc
    if is_catch:
        # Determine if either fighter has appeared in female (non-catch) bouts
        df_norm = dfc[~dfc.get("WEIGHTCLASS", pd.Series("")).astype(str).str.contains("Catch", na=False)]
        # If dfc lacks WEIGHTCLASS, fall back to bout-level female/male columns
        if df_norm.empty:
            df_norm = dfc.copy()
        # Any evidence that a fighter is female?
        def was_female(name: str) -> bool:
            if "female" in df_norm.columns:
                sel = (df_norm["name f1"].astype(str).str.strip() == name) | (
                    df_norm["name f2"].astype(str).str.strip() == name
                )
                return bool((df_norm.loc[sel, "female"] == 1).any())
            # Otherwise, infer via WEIGHTCLASS containing 'Women'
            if "WEIGHTCLASS" in df_norm.columns:
                sel = (df_norm["name f1"].astype(str).str.strip() == name) | (
                    df_norm["name f2"].astype(str).str.strip() == name
                )
                return bool(df_norm.loc[sel, "WEIGHTCLASS"].astype(str).str.contains("Women", na=False).any())
            return False

        n1_f = was_female(n1)
        n2_f = was_female(n2)

        if not n1_f and not n2_f:
            # If both are completely new (not in dfc) AND catch => default male with warning
            in_dfc1 = ((dfc["name f1"].astype(str).str.strip() == n1) | (dfc["name f2"].astype(str).str.strip() == n1)).any()
            in_dfc2 = ((dfc["name f1"].astype(str).str.strip() == n2) | (dfc["name f2"].astype(str).str.strip() == n2)).any()
            if not in_dfc1 and not in_dfc2:
                logger.warning(f"Catch-weight debut for both fighters; defaulting sex to male: {n1} vs {n2}")
                return 1, 0
        # If any is female, it's a women's bout
        if n1_f or n2_f:
            return 0, 1
        # Otherwise, male
        return 1, 0

    # Non-women non-catch => male
    return 1, 0


def clean_pred() -> pd.DataFrame:
    raw_path = get_data_path("raw") / "pred_raw.csv"
    out_path = get_data_path("interim") / "pred_clean.csv"

    dfr = open_csv(raw_path)
    if dfr.empty:
        raise FileNotFoundError(f"Input file not found or empty: {raw_path}")

    # Normalize/strip
    exp_cols = ["FIGHTER", "OPPONENT", "WEIGHTCLASS", "DATE"]
    for col in exp_cols:
        if col not in dfr.columns:
            raise KeyError(f"Missing required column '{col}' in pred_raw.csv")
    for col in ["FIGHTER", "OPPONENT", "WEIGHTCLASS", "DATE"]:
        dfr[col] = dfr[col].astype(str).str.strip()

    # Load references
    dfc = _load_cleaned()
    alt_map = _load_alt_spellings()
    mapper = NameMapper(dfc, alt_map)
    # Also load fighter totals for new fighters
    dft = _load_fighter_totals()
    dft_idx = _build_totals_index(dft)

    # Map names and compute tau
    names1, names2, new1, new2, taus = [], [], [], [], []

    # Special resolver for ambiguous Bruno Silva
    def _resolve_bruno_name(weightclass: str) -> str:
        wc = str(weightclass).strip().lower()
        if ("fly" in wc) or ("bantam" in wc):
            return "Bruno Fly Silva"
        elif "middle" in wc:
            return "Bruno Middle Silva"
        # Any other weightclass => raise
        raise ValueError('Change "Bruno Silva" to either "Bruno Fly Silva" or "Bruno Middle Silva"')

    for idx, row in dfr.iterrows():
        raw_f = str(row["FIGHTER"]).strip()
        raw_o = str(row["OPPONENT"]).strip()
        wc = str(row["WEIGHTCLASS"]).strip()

        n1, is_new1 = mapper.map_name(raw_f)
        n2, is_new2 = mapper.map_name(raw_o)

        # Override mapping for ambiguous Bruno Silva based on weightclass
        if raw_f == "Bruno Silva":
            n1 = _resolve_bruno_name(wc)
            is_new1 = True
        if raw_o == "Bruno Silva":
            n2 = _resolve_bruno_name(wc)
            is_new2 = True

        names1.append(n1)
        names2.append(n2)
        new1.append(is_new1)
        new2.append(is_new2)
        taus.append(_compute_tau(row["DATE"]))

    dfo = pd.DataFrame({
        "name f1": names1,
        "name f2": names2,
        "tau": taus,
        "WEIGHTCLASS": dfr["WEIGHTCLASS"].values,
    })

    # Weightclass one-hots (copying logic from clean_raw_data)
    # Note: classes are matched via substring contains
    classes = ["Catch ", "Straw", "Fly", "Bantam", "Feather", "Light", "Welter", "Middle", "Light Heavy"]
    wclasses = [w + "weight" for w in classes] + ["Heavyweight"]

    wc_series = dfo["WEIGHTCLASS"].astype(str)
    for wc in classes:
        dfo[f"{wc}weight"] = (wc_series.str.contains(wc, na=False)).astype(int)
    # Heavyweight is any 'Heavy' but not 'Light Heavy'
    dfo["Heavyweight"] = (
        wc_series.str.contains("Heavy", na=False) & ~wc_series.str.contains("Light Heavy", na=False)
    ).astype(int)

    # Bout flags (title vs normal)
    dfo["title bout"] = wc_series.str.contains("Title", na=False).astype(int)
    dfo["normal bout"] = 1 - dfo["title bout"]

    # Rounds flags
    dfo["5 rounds"] = dfo["title bout"].astype(int)  # title => 5 rounds
    dfo.loc[:, "3 rounds"] = 0
    dfo.loc[dfo["5 rounds"] == 0, "3 rounds"] = 1  # default others to 3 rounds
    if len(dfo) > 0:
        # First fight is always 5 rounds (main event), even if not title
        dfo.loc[dfo.index[0], ["5 rounds", "3 rounds"]] = [1, 0]

    # Sex flags
    male_f, female_f = [], []
    for idx, row in dfo.iterrows():
        m, f = _infer_sex_for_bout(dfc, row["name f1"], row["name f2"], row["WEIGHTCLASS"])
        male_f.append(m)
        female_f.append(f)
    dfo["male"], dfo["female"] = male_f, female_f

    # Height/Reach per fighter (last known from cleaned data first)
    h1 = [
        _last_known_value_for_fighter(dfc, n1, "height") for n1 in dfo["name f1"].values
    ]
    h2 = [
        _last_known_value_for_fighter(dfc, n2, "height") for n2 in dfo["name f2"].values
    ]
    r1 = [
        _last_known_value_for_fighter(dfc, n1, "reach") for n1 in dfo["name f1"].values
    ]
    r2 = [
        _last_known_value_for_fighter(dfc, n2, "reach") for n2 in dfo["name f2"].values
    ]

    # Age per fighter: extrapolate from last known age by tau difference (fallback to DOB for new fighters below)
    ages1, ages2 = [], []
    for idx, row in dfo.iterrows():
        tau_pred = int(row["tau"]) if not pd.isna(row["tau"]) else np.nan
        a1, t1 = _last_known_age_and_tau(dfc, row["name f1"])
        a2, t2 = _last_known_age_and_tau(dfc, row["name f2"])
        if not (pd.isna(a1) or pd.isna(t1) or pd.isna(tau_pred)):
            ages1.append(int(round(a1 + (tau_pred - t1))))
        else:
            ages1.append(np.nan)
        if not (pd.isna(a2) or pd.isna(t2) or pd.isna(tau_pred)):
            ages2.append(int(round(a2 + (tau_pred - t2))))
        else:
            ages2.append(np.nan)

    # For new fighters, try to fill height/reach/age from fighter totals; warn if not found
    warned_missing = set()
    for i, row in dfo.iterrows():
        tau_pred = row["tau"]

        if new1[i]:
            s = _lookup_totals_row(dft_idx, row["name f1"])
            if s is not None:
                ht, rc, dob = _extract_totals_metrics(s)
                if pd.isna(h1[i]) and not pd.isna(ht):
                    h1[i] = ht
                if pd.isna(r1[i]) and not pd.isna(rc):
                    r1[i] = rc
                if pd.isna(ages1[i]) and dob is not None and not pd.isna(tau_pred):
                    age_calc = _age_from_dob_at_tau(dob, int(tau_pred))
                    if age_calc is not None:
                        ages1[i] = age_calc
            else:
                key = f"NEW:{row['name f1']}"
                if key not in warned_missing:
                    logger.warning(f"New fighter not found in ufc_fighter_tott_clean.csv: {row['name f1']}")
                    warned_missing.add(key)

        if new2[i]:
            s = _lookup_totals_row(dft_idx, row["name f2"])
            if s is not None:
                ht, rc, dob = _extract_totals_metrics(s)
                if pd.isna(h2[i]) and not pd.isna(ht):
                    h2[i] = ht
                if pd.isna(r2[i]) and not pd.isna(rc):
                    r2[i] = rc
                if pd.isna(ages2[i]) and dob is not None and not pd.isna(tau_pred):
                    age_calc = _age_from_dob_at_tau(dob, int(tau_pred))
                    if age_calc is not None:
                        ages2[i] = age_calc
            else:
                key = f"NEW:{row['name f2']}"
                if key not in warned_missing:
                    logger.warning(f"New fighter not found in ufc_fighter_tott_clean.csv: {row['name f2']}")
                    warned_missing.add(key)

    dfo["height f1"], dfo["height f2"] = h1, h2
    dfo["reach f1"], dfo["reach f2"] = r1, r2
    dfo["age f1"], dfo["age f2"] = ages1, ages2

    # Select and order output columns
    weight_cols = wclasses  # already matches names created above
    base_cols = [
        "name f1",
        "name f2",
        "tau",
        "3 rounds",
        "5 rounds",
        "title bout",
        "normal bout",
        "male",
        "female",
    ]
    fighter_cols = ["age f1", "age f2", "height f1", "height f2", "reach f1", "reach f2"]
    # Some columns might not exist depending on input; guard with presence check
    ordered = [c for c in base_cols if c in dfo.columns] + \
              [c for c in weight_cols if c in dfo.columns] + \
              [c for c in fighter_cols if c in dfo.columns]

    dfo_out = dfo[ordered].copy()

    store_csv(dfo_out, out_path)
    logger.info(f"Saved {len(dfo_out)} upcoming fights to {out_path}")
    return dfo_out


if __name__ == "__main__":
    process_predictions()
