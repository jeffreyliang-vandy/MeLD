"""
One-time static audit of the preprocessed HDF5 used for VAE training.

Cheap sanity checks that eliminate the "the data already contains a NaN" branch
before spending hours reproducing the training NaN:

  * non-finite counts per dataset and per feature column
  * `cats` column value ranges vs their declared cardinality (`cards`)
    -> an out-of-range index makes F.cross_entropy in auto_loss produce NaN/assert
  * `bins` columns really in {0, 1}
  * non-finite `processed_data` positions that coincide with `missing == 1`
    -> auto_loss does `mse = mse * missing_nums`; multiplying a NaN by 0 does
       NOT clear it, and compute_sine_cosine would smear sin(NaN) everywhere

Timing note: every training epoch touches every row, so a *static* data NaN would
trip at epoch 0, not "after some epochs" -- expect this to come back clean. Run
it anyway; it is 30 seconds and the result is decisive either way.

    python MeLD/scripts/audit_h5.py -DP /path/to/data.h5
"""

import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", "-DP", required=True)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    with h5py.File(args.data_path, "r") as f:
        n_bins = int(f.attrs["n_bins"])
        n_cats = int(f.attrs["n_cats"])
        n_nums = int(f.attrs["n_nums"])
        cards = list(np.asarray(f.attrs["cards"]).tolist())
        print(f"n_bins={n_bins}  n_cats={n_cats}  n_nums={n_nums}  cards={cards}")

        dsets = ["processed_data", "time_info", "missing", "masking"]
        present = [d for d in dsets if d in f]
        N = f["processed_data"].shape[0]
        F = f["processed_data"].shape[-1]

        nonfinite = {d: 0 for d in present}
        col_bad = np.zeros(F, dtype=np.int64)
        col_min = np.full(F, np.inf)
        col_max = np.full(F, -np.inf)
        bins_out_of_01 = 0
        coincide_missing_nan = 0

        for start in range(0, N, args.chunk):
            sl = slice(start, min(N, start + args.chunk))
            arrs = {d: np.asarray(f[d][sl]) for d in present}

            for d in present:
                a = arrs[d]
                if np.issubdtype(a.dtype, np.floating):
                    nonfinite[d] += int((~np.isfinite(a)).sum())

            pd = arrs["processed_data"].astype(np.float64)
            flat = pd.reshape(-1, F)
            bad = ~np.isfinite(flat)
            col_bad += bad.sum(0)
            with np.errstate(all="ignore"):
                good = np.where(np.isfinite(flat), flat, np.nan)
                cmin = np.nanmin(good, axis=0)
                cmax = np.nanmax(good, axis=0)
            col_min = np.fmin(col_min, np.nan_to_num(cmin, nan=np.inf))
            col_max = np.fmax(col_max, np.nan_to_num(cmax, nan=-np.inf))

            if n_bins > 0:
                b = flat[:, :n_bins]
                bins_out_of_01 += int(
                    ((b != 0) & (b != 1) & np.isfinite(b)).sum()
                )

            if "missing" in arrs and arrs["missing"].shape[-1] == n_nums:
                miss = arrs["missing"].reshape(-1, n_nums) > 0.5
                nums_bad = bad[:, n_bins + n_cats: n_bins + n_cats + n_nums]
                coincide_missing_nan += int((nums_bad & miss).sum())

        print("\n-- non-finite counts per dataset --")
        for d, c in nonfinite.items():
            print(f"  {d:16s}: {c}")

        print("\n-- processed_data non-finite per column --")
        nz = np.nonzero(col_bad)[0]
        print(f"  columns with any non-finite: {nz.tolist() or 'none'}")
        if nz.size:
            print(f"  counts: {col_bad[nz].tolist()}")

        print("\n-- processed_data column ranges --")
        print(f"  min: {np.round(col_min, 4).tolist()}")
        print(f"  max: {np.round(col_max, 4).tolist()}")

        if n_cats > 0:
            print("\n-- cats columns vs cards --")
            for i in range(n_cats):
                lo = col_min[n_bins + i]
                hi = col_max[n_bins + i]
                ok = np.isfinite(lo) and lo >= 0 and hi < cards[i]
                print(f"  cat{i}: min={lo}  max={hi}  card={cards[i]}  "
                      f"{'OK' if ok else '*** OUT OF RANGE ***'}")

        print(f"\nbins values not in {{0,1}} : {bins_out_of_01}")
        print(f"non-finite num positions where missing==1 : {coincide_missing_nan}")
        print("\n(clean = every count above is 0 and all cats are OK)")


if __name__ == "__main__":
    main()
