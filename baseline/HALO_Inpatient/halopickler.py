import argparse
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import DP as dp  # Archer internal lib for partitioning

######################################################################
# Pickler: forward ↔ reverse utility                                 #
######################################################################

class Pickler:
    """Bidirectional helper for Archer longitudinal datasets.

    ➜ **Forward**: raw DataFrame → dummy/bin encode → `partition_multi_seq` →
      **package** (dict) → pickle.

    ➜ **Reverse**: pickle → package → flatten → inverse dummy → original‑shape
      *DataFrame* with **exact** original column order (minus `patient_id`).
    """

    ##################################################################
    # Init / state                                                   #
    ##################################################################
    def __init__(self):
        self._reverse_meta: Dict[str, Any] = {}
        self._feature_cols: List[str] = []
        self._orig_order: List[str] = []
        # Forward artifacts (set by fit)
        self.processed_data = None
        self.masking = None
        self.labels = None

    ##################################################################
    # Dummy expansion                                                #
    ##################################################################
    @staticmethod
    def _dummy_expand(
        df: pd.DataFrame,
        unique_threshold: int = 30,
        bins: int = 30,
        include_nan_bucket: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """One‑hot/quantise non‑binary numeric cols and collect metadata."""
        df_exp = df.copy()
        meta: Dict[str, Any] = {
            "dummy_to_original": {},
            "original_to_dummylist": {},
            "bin_intervals": {},
            "col_min_max": {},
        }

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            uniq = df[col].nunique(dropna=True)
            col_min, col_max = df[col].min(), df[col].max()
            meta["col_min_max"][col] = (col_min, col_max)

            # leave binary cols untouched
            if uniq <= 2:
                continue

            # ------------ small‑cardinality: direct category dummies ---------
            if uniq < unique_threshold:
                dummies = (
                    pd.get_dummies(df[col], prefix=col, dummy_na=include_nan_bucket)
                    .astype(int)
                )
                for dum in dummies.columns:
                    meta["dummy_to_original"][dum] = col
                    meta["bin_intervals"][dum] = None  # direct category flag
                meta["original_to_dummylist"].setdefault(col, []).extend(dummies.columns)
                df_exp = df_exp.drop(columns=[col]).join(dummies)
                continue

            # ------------ large‑cardinality: discretise then dummy ----------
            cats = pd.cut(df[col], bins=bins, duplicates="drop")
            dummies = (
                pd.get_dummies(cats, prefix=col, dummy_na=include_nan_bucket)
                .astype(int)
            )
            # map interval string → bounds
            interval_map = {
                str(iv): (float(iv.left), float(iv.right)) for iv in cats.cat.categories
            }
            for dum in dummies.columns:
                meta["dummy_to_original"][dum] = col
                if dum.endswith("nan"):
                    meta["bin_intervals"][dum] = (np.nan, np.nan)
                else:
                    iv_str = dum[len(col) + 1 :]
                    meta["bin_intervals"][dum] = interval_map.get(iv_str, (col_min, col_max))
            meta["original_to_dummylist"].setdefault(col, []).extend(dummies.columns)
            df_exp = df_exp.drop(columns=[col]).join(dummies)

        return df_exp, meta

    ##################################################################
    # Forward pipeline                                               #
    ##################################################################
    def fit(
        self,
        real: pd.DataFrame,
        pid_col: str,
        labels: pd.DataFrame,
        threshold: int = 1,
        visit_len: int = 120,
        include_nan_bucket: bool = True,
    ) -> "Pickler":
        """Encode + partition; keeps metadata for perfect inversion."""
        real = real.copy()
        labels = labels.copy()

        pid_series = real.pop(pid_col)
        _ = labels.pop(pid_col)

        # record original column order after pid removal
        self._orig_order = real.columns.tolist()

        # dummy expansion
        real_exp, self._reverse_meta = self._dummy_expand(
            real, include_nan_bucket=include_nan_bucket
        )
        self._feature_cols = real_exp.columns.tolist()
        real_exp[pid_col] = pid_series

        # partition to tensors
        tensors, _, masking = dp.partition_multi_seq(
            real_exp, threshold, pid_col, visit_len
        )
        self.processed_data = tensors.cpu().numpy()
        self.masking = masking.cpu().numpy()
        self.labels = labels.values
        return self

    def transform(self) -> Dict[str, Any]:
        """Return serialisable package with all metadata + records."""
        records = {}
        for pid in range(self.processed_data.shape[0]):
            valid_mask = self.masking[pid, :, -1] < 1  # 1 = padding
            seq = self.processed_data[pid][valid_mask]
            visits = [np.where(seq[v] > 0)[0].tolist() for v in range(seq.shape[0])]
            records[pid] = {"visits": visits, "labels": self.labels[pid]}

        return list(records.values())

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform()

    ##################################################################
    # Reverse pipeline                                               #
    ##################################################################
    def reverse(
        self,
        package_path: str,
        dtype: np.dtype | str = np.float32,
        rng_seed: int | None = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Unpickle package → reconstruct original DataFrame and labels."""
        with open(package_path, "rb") as fh:
            data = pickle.load(fh)

        # Flatten visits
        F = len(self._feature_cols)
        total_visits = sum(len(p["visits"]) for p in data)
        mat = np.zeros((total_visits, F), dtype=dtype)
        pid = np.zeros((total_visits,),dtype=int)
        labels = []
        r = 0
        id = 0
        for p in data:
            for visit in p["visits"]:
                mat[r, visit] = 1
                pid[r] = id
                r += 1
            labels.append(p["labels"])
            id += 1
        labels_arr = np.vstack(labels)

        df_dummy = pd.DataFrame(mat, columns=self._feature_cols)
        real_df = self._invert_dummy(df_dummy, rng_seed)
        return real_df, labels_arr, pid

    ##################################################################
    # Dummy inversion helper                                         #
    ##################################################################
    def _invert_dummy(self, df_dummy: pd.DataFrame, rng_seed: int | None) -> pd.DataFrame:
        """Efficiently invert dummy DataFrame back to original space.

        Avoids many `DataFrame.__setitem__` calls to keep frame un‑fragmented.
        Returns a **new** DataFrame assembled via `pd.concat`, then re‑indexed to
        the original column order stored in `self._orig_order`.
        """
        meta = self._reverse_meta
        rng = np.random.default_rng(rng_seed)

        # ----------- untouched columns (never expanded) ----------------
        untouched_cols = [c for c in self._feature_cols if c not in meta["dummy_to_original"]]
        parts: List[pd.DataFrame] = [df_dummy[untouched_cols].copy()]

        # ----------- reconstruct expanded numeric / cat columns --------
        for orig, dummies in meta["original_to_dummylist"].items():
            sub = df_dummy[dummies].values
            chosen = (sub > 0.5).argmax(axis=1)
            values: List[Any] = []
            for r, idx in enumerate(chosen):
                if sub[r, idx] <= 0.5:  # all zeros – NaN result
                    values.append(np.nan)
                    continue
                dum = dummies[idx]
                bin_info = meta["bin_intervals"].get(dum)

                # NaN bucket determination
                if (bin_info is None and dum.endswith("nan")) or (
                    bin_info is not None and np.isnan(bin_info[0]) and np.isnan(bin_info[1])
                ):
                    values.append(np.nan)
                    continue

                if bin_info is None:  # small‑cat category
                    raw = dum[len(orig) + 1 :]
                    try:
                        values.append(float(raw))
                    except ValueError:
                        values.append(raw)
                else:  # interval bucket
                    left, right = bin_info
                    values.append(rng.uniform(left, right))
            parts.append(pd.DataFrame({orig: values}, index=df_dummy.index))

        # ----------- combine & order -----------------------------------
        full = pd.concat(parts, axis=1)
        return full.loc[:, self._orig_order]

    @property
    def index_to_code(self): return self._index_to_code

    @property
    def code_to_index(self): return self._code_to_index


# ----------------------------------------------------------------------
# CLI (minimal) --------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    fwd = sub.add_parser("forward")
    fwd.add_argument("--csv")
    fwd.add_argument("--out")
    fwd.add_argument("--pid", default="patient_id")

    rev = sub.add_parser("reverse")
    rev.add_argument("--pkg")
    rev.add_argument("--out")

    args = parser.parse_args()

    if args.cmd == "forward":
        df = pd.read_csv(args.csv)
        pk = Pickler()
        pkg = pk.fit_transform(df, args.pid, df[[args.pid]].copy())
        pickle.dump(pkg, open(args.out, "wb"))
    else:
        pk = Pickler()
        real, _ = pk.reverse(args.pkg)
        real.to_csv(args.out, index=False)
