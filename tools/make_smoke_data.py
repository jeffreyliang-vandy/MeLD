"""Generate a tiny synthetic HDF5 dataset for smoke-testing the MeLD pipeline.

`0_data_preprocess.ipynb` shows how to build the four tensors but stops before writing
the HDF5 file that `1_train_vae.py` actually consumes. This closes that gap, so the
whole pipeline can be exercised end to end without touching real cohort data.

    python tools/make_smoke_data.py --out-dir /tmp/smoke

Writes `smoke.h5` (datasets processed_data/time_info/missing/masking; attrs
n_bins/n_cats/n_nums/cards) and `parser.pkl` (the fitted DataFrameParser step 4 needs).
"""
import argparse
import os
import pickle

import h5py
import numpy as np
import pandas as pd

from model import DP as dp

SEED = 20260818


def build_frame(n_patients, max_visits, rng):
    """A visit-level frame with binary, categorical and numeric columns.

    Column typing is decided by DataFrameParser.fit via cardinality, so the unique
    counts below are load-bearing: <=2 uniques become binary, 3..25 categorical, and
    anything above that numeric. See model/process_edited.py:238.
    """
    rows = []
    for pid in range(n_patients):
        n_visits = int(rng.integers(2, max_visits + 1))
        start = np.datetime64("2010-01-01") + rng.integers(0, 3000).astype("timedelta64[D]")
        for v in range(n_visits):
            rows.append(
                {
                    "patient_id": pid,
                    "date": start + np.timedelta64(int(v * rng.integers(30, 200)), "D"),
                    "sex": int(rng.integers(0, 2)),            # 2 uniques  -> binary
                    "on_art": int(rng.integers(0, 2)),         # 2 uniques  -> binary
                    "regimen": int(rng.integers(0, 5)),        # 5 uniques  -> categorical
                    "cd4": float(rng.normal(500, 150)),        # continuous -> numeric
                    "viral_load": float(rng.lognormal(6, 1.5)),
                    "weight": float(rng.normal(70, 12)),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["patient_id", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-patients", type=int, default=40)
    ap.add_argument("--max-visits", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = build_frame(args.n_patients, args.max_visits, rng)
    print(f"frame: {len(df)} rows, {df.patient_id.nunique()} patients")

    (processed, time_info, missing, masking), parser = dp.partition_multi_seq(
        df, threshold=args.threshold, column_to_partition="patient_id",
        max_len=args.max_visits,
    )

    # Must be read AFTER partition_multi_seq: it drops the partition column from the
    # parser and re-slices `missing` to match, so reading it earlier is off by one and
    # trips the n_nums == missing.shape[2] assertion in 1_train_vae.py.
    info = parser.datatype_info()
    n_bins, n_cats, n_nums = info["n_bins"], info["n_cats"], info["n_nums"]
    cards = info["cards"]

    print(f"tensors: processed={tuple(processed.shape)} time={tuple(time_info.shape)} "
          f"missing={tuple(missing.shape)} masking={tuple(masking.shape)}")
    print(f"types:   n_bins={n_bins} n_cats={n_cats} n_nums={n_nums} cards={cards}")

    assert n_bins + n_cats + n_nums == processed.shape[2], (
        f"n_bins+n_cats+n_nums={n_bins + n_cats + n_nums} != F={processed.shape[2]}")
    assert n_nums == missing.shape[2], (
        f"n_nums={n_nums} != missing.shape[2]={missing.shape[2]}")
    assert time_info.shape[2] % 2 == 0, "time_dim must be an even number of sin/cos pairs"
    assert n_bins > 0 and n_cats > 0 and n_nums > 0, (
        "smoke data must exercise all three column types")

    h5_path = os.path.join(args.out_dir, "smoke.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("processed_data", data=np.asarray(processed, dtype=np.float32))
        f.create_dataset("time_info", data=np.asarray(time_info, dtype=np.float32))
        f.create_dataset("missing", data=np.asarray(missing, dtype=np.float32))
        f.create_dataset("masking", data=np.asarray(masking, dtype=np.float32))
        f.attrs["n_bins"] = n_bins
        f.attrs["n_cats"] = n_cats
        f.attrs["n_nums"] = n_nums
        f.attrs["cards"] = np.asarray(cards, dtype=np.int64)

    parser_path = os.path.join(args.out_dir, "parser.pkl")
    with open(parser_path, "wb") as f:
        pickle.dump(parser, f)

    # A cohort table for the conditional (CFG) path: one row per sequence, row-aligned
    # with the latents that step 2 emits.
    cohort = (
        df.groupby("patient_id")
        .agg(sex=("sex", "max"), regimen=("regimen", "max"))
        .reset_index(drop=True)
    )
    cohort_path = os.path.join(args.out_dir, "cohort.parquet")
    cohort.to_parquet(cohort_path)

    print(f"\nwrote {h5_path}\n      {parser_path}\n      {cohort_path}")


if __name__ == "__main__":
    main()
