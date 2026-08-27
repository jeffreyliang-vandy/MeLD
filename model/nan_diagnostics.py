"""
Reusable NaN / divergence diagnostics for the DeapStack VAE training scripts.

This is a *trajectory recorder with a rolling replay buffer*, not merely a
"detect-a-NaN-and-stop" hook. The forward pass of the model has almost no
unbounded path from clean inputs (GRU gates bound hidden states to [-1, 1],
`fc_logvar` is Hardtanh(-6, 2), the `nums` head is sigmoid-bounded, `sin`/`cos`
are bounded regardless of angle). So a NaN that appears only after some epochs
almost certainly comes from parameters drifting to +/-inf through the optimizer.
To diagnose that we need the step *before* the NaN and the offending batch, not
just the moment the NaN surfaces.

Usage (single-GPU, see 1_train_vae_debug.py):

    from MeLD.model import nan_diagnostics as nd
    from MeLD.model import timeautoencoder as tae

    nd.set_global_seed(args.seed)
    diag = nd.NaNDiagnostics(model, optimizer, out_dir, buffer_steps=200,
                             anomaly=args.anomaly,
                             inject_nan_at_step=args.inject_nan_at_step)
    diag.attach_forward_hooks()
    diag.patch_sine_cosine(tae)

    # per training step:
    diag.begin_step()
    diag.maybe_inject(global_step)
    diag.check_inputs({"data": data, "time_info": time_info,
                       "missing": missing, "masking": masking})
    optimizer.zero_grad()
    RE, KL = model.get_loss(data, time_info, missing, masking)
    diag.check_loss(RE, KL)
    loss = RE + beta * torch.maximum(KL, delta)
    loss.backward()
    diag.check_grads()
    diag.optimizer_state_stats()
    diag.snapshot(global_step, batch, idx)      # BEFORE optimizer.step()
    optimizer.step()
    diag.check_params()
    diag.finish_step(global_step, epoch, scalars, batch, idx)  # records or trips

On the first non-finite value `finish_step` writes, into `out_dir`:
    diagnosis.log       - single human-readable log (config + heartbeats + the
                          full trip summary); copy THIS file off the server, or
                          the whole `out_dir/` for the binary dumps too
    nan_report.json     - step/epoch, first bad stage, per-stage findings, indices
    metrics.jsonl       - coarse trace, one record every `metrics_every` steps
                          (whole run)
    metrics_tail.jsonl  - FULL per-step records for the last `buffer_steps` steps
    batch.pt            - the offending input tensors + sample indices (CPU)
    model_at_trip.pt / optim_at_trip.pt   - live (post-update) state
    snapshot_step<N>.pt - the rolling pre-update model+optimizer snapshots
and raises `NaNTripped` so the driver can stop cleanly.

Instrumentation cost (diagnostic, not production): the leaf-module finiteness
hooks force ~one GPU sync per module per step, and `snapshot()` deep-copies the
model + optimizer state to CPU every step. Expect the run to be materially slower
than `1_train_vae.py`; `--heartbeat_every` / `--metrics_every` control log volume,
not this overhead.

CAVEAT - reproducing the ORIGINAL NaN: the debug drivers change determinism
(cuDNN deterministic, benchmark off), and default to a fixed seed + deterministic
split. Any of these can shift *where* the NaN lands (a different epoch) or, less
likely, mask it. A NaN at a different epoch is still a valid diagnosis. If it does
not trip within a comparable number of epochs, re-run with `--nondeterministic`
and `--num_workers 8` to get closer to the original numerics.
"""

import os
import json
import copy
import math
import random
import datetime
from collections import deque

import numpy as np
import torch

STAGES = ("inputs", "forward", "RE", "KL", "grads", "params")


class NaNTripped(RuntimeError):
    """Raised after dumps are written, on the first non-finite value seen."""

    def __init__(self, step, stage, detail):
        super().__init__(
            f"[NaNDiagnostics] non-finite at step {step}, stage '{stage}': {detail}"
        )
        self.step = step
        self.stage = stage
        self.detail = detail


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG we can reach and (optionally) force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # very old torch
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    return seed


# ---------------------------------------------------------------------------
# small tensor helpers
# ---------------------------------------------------------------------------
def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)


def _tensor_report(t):
    """finite/min/max/absmax summary for a float tensor, else None."""
    if not torch.is_tensor(t) or not t.is_floating_point():
        return None
    with torch.no_grad():
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum().item())
        rep = {"n_bad": n_bad, "numel": int(t.numel())}
        if finite.any():
            good = t[finite]
            rep["min"] = float(good.min().item())
            rep["max"] = float(good.max().item())
            rep["absmax"] = float(good.abs().max().item())
        return rep


def _to_cpu(obj):
    """Deep copy an arbitrary (possibly nested) structure onto the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


def finite_float(x):
    try:
        v = float(x.detach().item()) if torch.is_tensor(x) else float(x)
    except Exception:
        return None, False
    return v, math.isfinite(v)


# ---------------------------------------------------------------------------
# main diagnostics object
# ---------------------------------------------------------------------------
class NaNDiagnostics:
    def __init__(
        self,
        model,
        optimizer,
        out_dir,
        *,
        buffer_steps: int = 200,
        snapshot_steps: int = 2,
        top_k_grad: int = 15,
        anomaly: bool = False,
        inject_nan_at_step: int = -1,
        inject_nan_phase: str = "pre_step",
        metrics_every: int = 25,
        is_main: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.out_dir = out_dir
        self.is_main = is_main
        self.top_k_grad = top_k_grad
        self.inject_nan_at_step = (
            inject_nan_at_step if inject_nan_at_step is not None else -1
        )
        self.inject_nan_phase = inject_nan_phase
        # metrics.jsonl is a coarse, long-run trace: write only every Nth step so
        # a multi-epoch run stays small. metrics_tail.jsonl (dumped on trip) keeps
        # FULL per-step resolution for the last `buffer_steps` steps.
        self.metrics_every = max(1, metrics_every)

        if is_main:
            os.makedirs(out_dir, exist_ok=True)
            self._metrics_fh = open(os.path.join(out_dir, "metrics.jsonl"), "a")
            self._log_fh = open(os.path.join(out_dir, "diagnosis.log"), "a")
        else:
            self._metrics_fh = None
            self._log_fh = None

        self.buffer = deque(maxlen=buffer_steps)
        self.snapshots = deque(maxlen=max(2, snapshot_steps))
        self._hooks = []
        self._orig_sine_cosine = None
        self._tae_mod = None

        if anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.begin_step()

    # -- per-step lifecycle -------------------------------------------------
    def begin_step(self):
        self._findings = {s: None for s in STAGES}
        self._fwd_bad = []
        self._sine_stats = {}
        self._latent_stats = {}
        self._grad_topk = None
        self._grad_global_norm = None
        self._weight_global_norm = None
        self._opt_stats = None

    # -- consolidated human-readable log --------------------------------
    def log(self, msg, echo=True):
        """Append one line to <out_dir>/diagnosis.log (and optionally stdout).

        Route the driver's own logging through here so a single file, copyable
        off the server, holds the whole diagnosis.
        """
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} | {msg}"
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()
        if echo:
            print(line)

    def heartbeat(self, step, epoch, every=1):
        """Compact one-liner into diagnosis.log every `every` steps."""
        if self._log_fh is None or (step % max(1, every)):
            return
        lat = self._latent_stats or {}
        sine = self._sine_stats or {}
        re_v = getattr(self, "_last_re", None)
        kl_v = getattr(self, "_last_kl", None)
        re_v = float("nan") if re_v is None else re_v
        kl_v = float("nan") if kl_v is None else kl_v
        self.log(
            "hb "
            f"step={step} ep={epoch} "
            f"RE={re_v:.4g} KL={kl_v:.4g} "
            f"gnorm={self._grad_global_norm} wnorm={self._weight_global_norm} "
            f"mu_absmax={lat.get('mu_absmax')} "
            f"logvar=[{lat.get('logvar_min')},{lat.get('logvar_max')}] "
            f"sat_hi={lat.get('logvar_frac_sat_high')} "
            f"sine_angle={sine.get('implied_max_angle')} "
            f"top_grad={(self._grad_topk or [{}])[0]}",
            echo=False,
        )

    # -- instrumentation setup -------------------------------------------------
    def attach_forward_hooks(self):
        """Finiteness hook on every leaf module + a stats hook on the Encoder."""
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if len(list(module.children())) == 0:
                self._hooks.append(
                    module.register_forward_hook(self._make_leaf_hook(name))
                )
            if module.__class__.__name__ == "Encoder":
                self._hooks.append(module.register_forward_hook(self._encoder_hook))
        return self

    def _make_leaf_hook(self, name):
        def hook(mod, inp, out):
            for t in _iter_tensors(out):
                if t.is_floating_point() and not torch.isfinite(t).all():
                    self._fwd_bad.append(
                        {
                            "module": name,
                            "cls": mod.__class__.__name__,
                            **(_tensor_report(t) or {}),
                        }
                    )
                    break

        return hook

    def _encoder_hook(self, mod, inp, out):
        try:
            emb, mu_z, logvar_z = out
        except Exception:
            return
        with torch.no_grad():
            self._latent_stats = {
                "mu_absmax": float(mu_z.abs().max().item()),
                "logvar_min": float(logvar_z.min().item()),
                "logvar_max": float(logvar_z.max().item()),
                "logvar_frac_sat_low": float(
                    (logvar_z <= -6.0 + 1e-4).float().mean().item()
                ),
                "logvar_frac_sat_high": float(
                    (logvar_z >= 2.0 - 1e-4).float().mean().item()
                ),
                "emb_absmax": float(emb.abs().max().item()),
            }

    def patch_sine_cosine(self, tae_module):
        """Wrap the free function `compute_sine_cosine` (no hook can see it).

        The backward path through it carries a `2**(num_terms-1) * pi` multiplier
        (~1e5 for num_terms=16) into `mlp_nums` - the top gradient-amplification
        suspect. We log the implied max angle and output finiteness.
        """
        self._tae_mod = tae_module
        self._orig_sine_cosine = tae_module.compute_sine_cosine
        orig = self._orig_sine_cosine

        def wrapped(v, num_terms):
            with torch.no_grad():
                v_absmax = (
                    float(v.detach().abs().max().item())
                    if torch.is_tensor(v)
                    else float("nan")
                )
            out = orig(v, num_terms)
            nt = int(num_terms.item() if torch.is_tensor(num_terms) else num_terms)
            self._sine_stats = {
                "v_absmax": v_absmax,
                "num_terms": nt,
                "implied_max_angle": (2.0 ** (nt - 1)) * math.pi * v_absmax,
                "out_finite": bool(torch.isfinite(out).all().item()),
            }
            if not self._sine_stats["out_finite"]:
                self._fwd_bad.append(
                    {
                        "module": "compute_sine_cosine",
                        "cls": "func",
                        **(_tensor_report(out) or {}),
                    }
                )
            return out

        tae_module.compute_sine_cosine = wrapped
        return self

    def restore(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._orig_sine_cosine is not None and self._tae_mod is not None:
            self._tae_mod.compute_sine_cosine = self._orig_sine_cosine
        if self._metrics_fh is not None:
            self._metrics_fh.flush()
        if self._log_fh is not None:
            self._log_fh.flush()

    # -- self-test ----------------------------------------------------------
    def maybe_inject(self, step, phase="pre_step"):
        """Poke a NaN into fc_mu.weight to confirm the harness trips + dumps.

        phase="pre_forward": injected before the forward pass -> the NaN
            propagates through activations, so the trip fires at stage 'forward'
            (naming encoder ...fc_mu.weight).
        phase="pre_step" (default): injected after check_grads and before
            optimizer.step() -> forward / RE / KL / grads are all still finite,
            so the trip fires at stage 'params'. This is the stage that matches
            the primary hypothesis (Adam-driven weight drift), so it exercises
            the more relevant code path.
        """
        if self.inject_nan_at_step is None or self.inject_nan_at_step < 0:
            return
        if step != self.inject_nan_at_step or phase != self.inject_nan_phase:
            return
        for name, p in self.model.named_parameters():
            if name.endswith("fc_mu.weight"):
                with torch.no_grad():
                    p.view(-1)[0] = float("nan")
                self.log(
                    f"self-test: injected NaN into {name}[0] at step {step} "
                    f"(phase={phase})"
                )
                return

    # -- tripwire stages --------------------------------------------------
    def check_inputs(self, named_tensors: dict):
        bad = []
        for k, t in named_tensors.items():
            if (
                torch.is_tensor(t)
                and t.is_floating_point()
                and not torch.isfinite(t).all()
            ):
                bad.append({"name": k, **(_tensor_report(t) or {})})
        if bad:
            self._findings["inputs"] = bad
        return bad

    def check_loss(self, re, kl):
        detail = {}
        re_v, re_ok = finite_float(re)
        kl_v, kl_ok = finite_float(kl)
        if not re_ok:
            detail["RE"] = re_v
            self._findings["RE"] = {"RE": re_v}
        if not kl_ok:
            detail["KL"] = kl_v
            self._findings["KL"] = {"KL": kl_v}
        self._last_re, self._last_kl = re_v, kl_v
        return detail

    def check_grads(self):
        rows = []
        total_sq = 0.0
        n_bad = 0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            finite = torch.isfinite(g)
            gb = int((~finite).sum().item())
            n_bad += gb
            gnorm = float(g[finite].norm().item()) if finite.any() else float("inf")
            if math.isfinite(gnorm):
                total_sq += gnorm * gnorm
            rows.append((name, gnorm, gb))
        rows.sort(
            key=lambda r: (r[1] if math.isfinite(r[1]) else float("inf")),
            reverse=True,
        )
        self._grad_topk = [
            {"param": n, "grad_norm": gn, "n_bad": gb}
            for n, gn, gb in rows[: self.top_k_grad]
        ]
        self._grad_global_norm = math.sqrt(total_sq) if n_bad == 0 else float("inf")
        if n_bad:
            self._findings["grads"] = {
                "n_bad_total": n_bad,
                "top": self._grad_topk[:5],
            }
        return self._grad_global_norm

    def check_params(self):
        bad = []
        total_sq = 0.0
        for name, p in self.model.named_parameters():
            d = p.detach()
            finite = torch.isfinite(d)
            pb = int((~finite).sum().item())
            if pb:
                bad.append({"param": name, **(_tensor_report(d) or {})})
            if finite.any():
                total_sq += float(d[finite].norm().item()) ** 2
        self._weight_global_norm = math.sqrt(total_sq) if not bad else float("inf")
        if bad:
            self._findings["params"] = bad
        return self._weight_global_norm

    def optimizer_state_stats(self):
        want = {r["param"] for r in (self._grad_topk or [])[:5]}
        stats = {}
        for name, p in self.model.named_parameters():
            if name not in want:
                continue
            st = self.optimizer.state.get(p, {})
            entry = {}
            for key in ("exp_avg", "exp_avg_sq"):
                t = st.get(key)
                if torch.is_tensor(t):
                    td = t.detach()
                    entry[key] = {
                        "min": float(td.min().item()),
                        "max": float(td.max().item()),
                        "mean": float(td.float().mean().item()),
                        "finite": bool(torch.isfinite(td).all().item()),
                    }
            if entry:
                stats[name] = entry
        self._opt_stats = stats
        return stats

    # -- rolling replay snapshot ----------------------------------------
    def snapshot(self, step, batch, indices):
        self.snapshots.append(
            {
                "step": step,
                "model_state": _to_cpu(self.model.state_dict()),
                "optim_state": _to_cpu(self.optimizer.state_dict()),
                "indices": _to_cpu(indices),
            }
        )

    # -- record / trip -------------------------------------------------
    def _build_record(self, step, epoch, scalars):
        rec = {"step": step, "epoch": epoch}
        if scalars:
            rec.update(scalars)
        rec.update(
            {
                "grad_global_norm": self._grad_global_norm,
                "weight_global_norm": self._weight_global_norm,
                "grad_topk": self._grad_topk,
                "opt_state": self._opt_stats,
                "sine": self._sine_stats or None,
                "latent": self._latent_stats or None,
                "forward_bad": self._fwd_bad or None,
            }
        )
        return rec

    def _write_record(self, rec):
        self.buffer.append(rec)  # full-resolution ring buffer, always
        if self._metrics_fh is not None and (rec["step"] % self.metrics_every == 0):
            self._metrics_fh.write(json.dumps(rec, default=str) + "\n")
            self._metrics_fh.flush()

    def finish_step(self, step, epoch, scalars, batch, indices, record=True):
        """Merge findings, record the step, and trip on the earliest bad stage."""
        if self._fwd_bad and self._findings["forward"] is None:
            self._findings["forward"] = list(self._fwd_bad)

        rec = self._build_record(step, epoch, scalars)
        if record:
            self._write_record(rec)

        for stage in STAGES:
            if self._findings[stage]:
                self.trip(step, epoch, stage, self._findings[stage],
                          batch, indices, rec)
        return rec

    def trip(self, step, epoch, stage, detail, batch, indices, rec):
        if self.is_main:
            d = self.out_dir
            report = {
                "step": step,
                "epoch": epoch,
                "first_bad_stage": stage,
                "stage_order": list(STAGES),
                "findings": {
                    s: self._findings[s] for s in STAGES if self._findings[s]
                },
                "indices": (
                    indices.detach().cpu().tolist()
                    if torch.is_tensor(indices)
                    else indices
                ),
                "record": rec,
            }
            with open(os.path.join(d, "nan_report.json"), "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            with open(os.path.join(d, "metrics_tail.jsonl"), "w") as fh:
                for r in self.buffer:
                    fh.write(json.dumps(r, default=str) + "\n")
            torch.save(
                {"batch": _to_cpu(batch), "indices": _to_cpu(indices)},
                os.path.join(d, "batch.pt"),
            )
            torch.save(_to_cpu(self.model.state_dict()),
                       os.path.join(d, "model_at_trip.pt"))
            torch.save(_to_cpu(self.optimizer.state_dict()),
                       os.path.join(d, "optim_at_trip.pt"))
            for snap in self.snapshots:
                torch.save(snap, os.path.join(d, f"snapshot_step{snap['step']}.pt"))

            self._write_trip_summary(step, epoch, stage, rec, report["indices"])
            print(
                f"[NaNDiagnostics] TRIP at step {step} (epoch {epoch}), "
                f"first bad stage='{stage}'. Dumps written to {d}"
            )
        raise NaNTripped(step, stage, detail)

    def _write_trip_summary(self, step, epoch, stage, rec, indices):
        """Plain-text, copy-friendly summary appended to diagnosis.log."""
        lines = []
        lines.append("=" * 78)
        lines.append(f"NaN TRIP  step={step}  epoch={epoch}")
        lines.append(f"first non-finite stage : {stage}   (order: {' -> '.join(STAGES)})")
        lines.append("-" * 78)
        for s in STAGES:
            f = self._findings.get(s)
            if f:
                lines.append(f"[{s}] {json.dumps(f, default=str)}")
        lines.append("-" * 78)
        lines.append(f"RE={rec.get('RE')}  KL={rec.get('KL')}  loss={rec.get('loss')}  "
                     f"beta={rec.get('beta')}")
        lines.append(f"grad_global_norm  = {rec.get('grad_global_norm')}")
        lines.append(f"weight_global_norm= {rec.get('weight_global_norm')}")
        if rec.get("latent"):
            lines.append(f"latent   : {json.dumps(rec['latent'], default=str)}")
        if rec.get("sine"):
            lines.append(f"sine_cos : {json.dumps(rec['sine'], default=str)}")
        lines.append("top grad-norm params:")
        for r in (rec.get("grad_topk") or [])[:10]:
            lines.append(f"    {r}")
        if rec.get("opt_state"):
            lines.append("adam moments (top grad params):")
            for k, v in rec["opt_state"].items():
                lines.append(f"    {k}: {json.dumps(v, default=str)}")
        lines.append("-" * 78)
        idx_str = indices if not isinstance(indices, list) else indices[:64]
        lines.append(f"offending sample indices (h5 rows): {idx_str}")
        lines.append("dumps in this dir:")
        for fn in ("nan_report.json", "metrics.jsonl", "metrics_tail.jsonl",
                   "batch.pt", "model_at_trip.pt", "optim_at_trip.pt"):
            lines.append(f"    {fn}")
        for snap in self.snapshots:
            lines.append(f"    snapshot_step{snap['step']}.pt  (pre-update model+optim)")
        lines.append("copy this whole directory off the server to analyse offline.")
        lines.append("=" * 78)
        self.log("\n".join(lines), echo=True)
