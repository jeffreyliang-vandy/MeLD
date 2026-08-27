# VAE NaN-loss — diagnosis report

Source: `diagnosis.log` (single-GPU `1_train_vae_debug.py`, seed 0, deterministic).
Run reproduced the failure: **NaN at global step 618, epoch 11**.

---

## 1. Verdict

**Unbounded gradients in the backward pass, with no gradient clipping anywhere.**

The forward pass is always finite. Adam's second-moment normalisation keeps the
*weights* bounded (`weight_global_norm` stays ≈ 272 for the entire run), so the
model limps along for ~600 steps while the *gradient* norm oscillates between
1e4 and 1e10. The NaN appears the moment one minibatch's raw gradient overflows
fp32 (> 3.4e38) inside `backward()` → `Inf` → `Inf × 0 = NaN` in the mask
multiplications of `auto_loss` → the NaN floods back through the **shared**
embedding sum `x_emb_sum` and poisons the whole `encoder.Emb` block.

The dominant gradient amplifier is the sinusoidal feature encoder
`compute_sine_cosine(x_nums, num_terms=16)` — frequency ladder up to `2**15`,
i.e. Jacobian entries up to `2**15 · π ≈ 1.03e5` per term, 16 terms. This is
compounded by an **under-regularised latent** (`mu` up to 8.5, `logvar` railed at
the Hardtanh ceiling `+2` for 53 % of entries, `emb` up to 16) and a post-LayerNorm
transformer block.

Confidence: **high** for the mechanism and the fix. The exact overflow step and
whether one specific batch/sample triggered it need the rest of the `nan_diag/`
dump (see §5).

---

## 2. Evidence from the log

### 2.1 Gradient explodes, weights do not

| step | epoch | `grad_global_norm` | `weight_global_norm` | top-grad param |
|-----:|------:|-------------------:|---------------------:|----------------|
| 0    | 0 | 1.1e1  | 274.1 | `decoder.decoder_mlp.2.weight` |
| 50   | 0 | 1.1e1  | 272.9 | `decoder.bins_linear.weight` |
| 100  | 1 | 3.2e0  | 271.5 | `decoder.bins_linear.weight` |
| 150  | 2 | **7.5e1**  | 270.7 | `encoder.Emb.mlp_output.2.weight` |
| 200  | 3 | **5.8e8**  | 271.8 | `encoder.Emb.mlp_output.2.weight` |
| 250  | 4 | 1.4e5  | 271.5 | `encoder.Emb.mlp_output.2.weight` |
| 300  | 5 | 9.6e4  | 271.4 | `encoder.Emb.mlp_output.2.weight` |
| 400  | 7 | 4.8e4  | 271.5 | `encoder.Emb.mlp_output.2.weight` |
| 500  | 9 | 3.3e5  | 272.0 | `encoder.Emb.mlp_output.2.weight` |
| 550  | 10 | **2.7e6**  | 272.3 | `encoder.Emb.mlp_output.2.weight` |
| 600  | 11 | **1.0e10** | 272.4 | `encoder.Emb.mlp_output.2.weight` |
| 618  | 11 | **inf → NaN** | inf (post-step) | — TRIP — |

- Weight norm flat → this is **not** slow optimiser drift; it is recurrent
  gradient explosion that Adam keeps absorbing (`update ≈ lr · m̂/√v̂`, bounded
  by ~`lr` no matter how big the gradient) until fp32 finally overflows.
- The consistently-largest gradient is `encoder.Emb.mlp_output.2.weight` — the
  last Linear of `Embedding_data`, i.e. right after the categorical embeddings +
  the `mlp_nums` (sinusoidal) branch are concatenated.

### 2.2 First non-finite stage = `grads` (not `forward`, not the loss)

```
NaN TRIP  step=618  epoch=11
first non-finite stage : grads   (order: inputs -> forward -> RE -> KL -> grads -> params)
RE=10.225  KL=4.195  loss=10.225  beta=0.0
grad_global_norm  = inf
weight_global_norm= inf          # check_params, AFTER optimizer.step applied the bad grads
```

Forward finite, `RE`/`KL` finite. The `Inf` is created **inside `backward()`**.

`beta = 0.0` at this step (cyclic β schedule at a cycle start) → the KL term
contributes ~nothing to the gradient *on this step*. The trigger is the
**reconstruction path through the embedding block**, not the KL term. (Weak KL
regularisation is still a background cause — see §2.4.)

### 2.3 Which parameters went NaN (`[params]` findings)

- `encoder.Emb.mlp_nums.0.weight` — **fully** NaN (230 400 / 230 400), also
  `mlp_nums.0.bias`, `mlp_nums.2.weight`, `mlp_nums.2.bias` fully NaN.
  `mlp_nums` is the MLP that consumes `compute_sine_cosine`'s output.
- **Every** `encoder.Emb.embeddings_list.*.weight` (all 263 tables) — exactly
  `n_bad = 128` each = **one full row** (`emb_dim = 128`) NaN. Those are the rows
  the offending batch indexed; the embedding's `weight.grad` is a scatter-add, so
  a NaN in the shared `x_emb_sum` writes NaN into exactly the touched rows.
- `encoder.Emb.mlp_output.*`, `encoder.Emb.paddings_list/missings_list.*`,
  `encoder.encoder_Transformer.conv_layer1.*` (dense → fully NaN),
  `...feature_layer.layers.0.self_attn.in_proj_*`.

This is precisely the fan-out of a NaN injected into `x_emb_sum` /
`final_emb = mlp_output(x_emb_sum)` and propagated backward.

### 2.4 Latent space is not regularised

```
latent : mu_absmax=8.51  logvar_min=-4.21  logvar_max=2.0
         logvar_frac_sat_high=0.526   emb_absmax=16.07
sine_cos: v_absmax=1.0  num_terms=16  implied_max_angle=102943.7   (= 2**15 · π)
```

- `fc_logvar` uses `nn.Hardtanh(min_val=-6., max_val=2.)`
  (`model/timeautoencoder.py:193/196`). **53 % of `logvar` entries are railed at
  the `+2` ceiling**, where Hardtanh's gradient is exactly 0 → the logvar head is
  effectively frozen and `std` sits at `exp(1) ≈ 2.7`.
- `mu_absmax = 8.5`, `emb_absmax = 16` for `lat_dim = 32`. With `max_beta = 0.01`
  and `min_kl = 0.0` (so `torch.maximum(KL, delta)` is a no-op), nothing pulls
  the latent back toward N(0, I). A large, unconstrained latent → large decoder
  activations → large decoder→encoder gradients.

### 2.5 The `num_terms = 16` sinusoidal encoder

`compute_sine_cosine` (`model/timeautoencoder.py:27`), called with
`num_terms = 16` (`:117`):

```
angles = 2**arange(16) * pi * v          # top frequency 2**15 · pi ≈ 1.03e5
out    = cat([sin(angles), cos(angles)]) # bounded in forward  (out_finite = true)
```

Forward is bounded, but the **Jacobian** `d out / d v` has entries
`± 2**k · pi · {cos,-sin}` — up to **1.03e5** for `k = 15`. Every gradient that
passes back through `mlp_nums` into this transform is multiplied by that ladder
and summed over 16 terms. Once the upstream gradient (encoder/decoder, amplified
by §2.4) is O(1e3–1e6), the product reaches fp32's ceiling. `mlp_nums.0.weight`
being the *first* thing fully NaN is consistent with the overflow originating at
the sinusoidal-feature boundary.

Also compounding: `auto_loss` sets `first_visit_scale = 10.0`
(`model/timeautoencoder.py:328`), multiplying the L=0 timestep's reconstruction
loss (and its gradient) by 11×.

---

## 3. Root cause (chain)

1. `compute_sine_cosine(num_terms=16)` puts frequencies up to `2**15` into the
   numeric-feature embedding → backward Jacobian up to ~1e5 per term.
2. Latent is unconstrained (`mu`≈8.5, `logvar` railed at Hardtanh `+2`,
   `emb`≈16; `max_beta` 0.01, `min_kl` 0) → large decoder activations → large
   gradients flowing back into the shared `encoder.Emb` block.
3. Post-LayerNorm `TransformerEncoderLayer` (`get_torch_trans`,
   `model/timeautoencoder.py:141`) adds early-training gradient spikes.
4. **No `clip_grad_norm_` in `1_train_vae.py` or `1_train_vae_aclr.py`.** Adam
   normalises the *weight update*, so weights stay bounded (norm ≈ 272) and
   training looks fine — but the *raw gradient* is free to grow. It oscillates
   1e4–1e10 for ~400 steps.
5. At step 618 one minibatch's backward gradient overflows fp32 → `Inf`.
   `Inf × 0` (the `real_mask` / `eos_mask` / `missing_nums` / `first_visit_mask`
   multiplications in `auto_loss`) → `NaN`.
6. The NaN is in `x_emb_sum` (shared by every embedding table + `mlp_nums` +
   `mlp_output` + the transformer). Backward scatters it into one row of all 263
   embedding tables and all of `mlp_nums`. Next step everything is NaN.

## 4. Why single-GPU is worse than the `accelerate` path

Not yet confirmed from data, but the consistent hypothesis:

- Single-GPU takes one optimiser step per 512-sample batch. `1_train_vae_aclr.py`
  uses `Accelerator(split_batches=True)` **and** `per_gpu = batch_size //
  num_processes` (batch divided twice), and all-reduce **averages** gradients
  across ranks — both smooth the extreme-gradient spikes and reduce the number
  of "fp32-overflow dice rolls" per epoch.
- To confirm: run `1_train_vae_aclr_debug.py` on 1 proc vs N procs and compare
  the recorded `actual_batch` and the step where `grad_global_norm` first
  exceeds ~1e6.

## 5. Additional data that would sharpen this (from the un-copied dump)

Only `diagnosis.log` was copied. The rest of `<checkpoint_dir>/nan_diag/` would
pin the last uncertainties — none change the fix:

- **`metrics_tail.jsonl`** — per-step (not per-50) `grad_global_norm` / `sine` /
  `latent` for steps ~420–618: is step 618 a sudden spike (one pathological
  batch) or the crest of the ramp?
- **`batch.pt`** — inspect `x_nums` for the offending 512 rows: any values
  pathologically close to 0 or 1 (where `cos(2**15 π v)` derivative is maximal)?
  any rare category?
- **`snapshot_step617.pt` vs `snapshot_step618.pt`** — diff
  `encoder.Emb.mlp_output.2.weight` and `mlp_nums.0.weight` magnitudes.
- Confirm whether the `accelerate` run trips at all / at which epoch.

---

## 6. Recommended fix

Do 1 + 2 together (minimal, stops the NaN); 3–5 address the underlying
instability so you are not one hyperparameter away from it returning.

1. **Gradient clipping.** In `1_train_vae.py` and `1_train_vae_aclr.py`,
   immediately before `optimizer.step()`:
   ```python
   torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)   # 1.0–5.0
   # accelerate: accelerator.clip_grad_norm_(ae.parameters(), 1.0)
   ```
   The log shows Adam already bounds the weights; clipping just stops the raw
   gradient from overflowing fp32.

2. **Cut `num_terms` in `compute_sine_cosine`** from 16 to ~6–8 (top frequency
   `2**5`–`2**7`). Frequencies far above the data's resolution are noise whose
   only effect is ~1e4–1e5 derivative magnitude. Update the `mlp_nums` input dim
   accordingly (`2 * num_terms * n_nums`, currently hard-coded `32 * n_nums`).

3. **Constrain the latent.**
   - Lower the `fc_logvar` Hardtanh ceiling (`max_val=2.` → `0.` or lower), or
     replace Hardtanh with a bound that keeps a gradient (`-6 + 8*sigmoid(x)` or
     `tanh`-based). 53 % railed entries = a dead head.
   - Make KL actually bite: raise `--max_beta`, and/or set a real free-bits floor
     (`--min_kl` > 0 so `torch.maximum(KL, delta)` engages). `mu_absmax=8.5` for
     `lat_dim=32` should not happen.

4. **`norm_first=True`** in `get_torch_trans` (`model/timeautoencoder.py:141`).
   Pre-LN transformers do not need warmup and produce far smaller early gradients.

5. Minor: `--lr 4e-4` is high for this stack without warmup — add a short LR
   warmup, and drop `first_visit_scale` (10.0 → 2–3).

---

## 7. Validation runs — §6.1 and §6.2 are NOT sufficient

Three instrumented runs, same seed 0, `batch_size=512`, `lr=4e-4`, `lat_dim=32`,
`max_beta=0.01`, cyclic β schedule.

| run | config | outcome | first bad stage |
|-----|--------|---------|-----------------|
| `diagnosis.log`  | baseline (`num_terms=16`, no clip)      | NaN @ **step 618 / epoch 11**  | `grads` |
| `diagnosis2.log` | `--sincos_num_terms 8` (fix 2 only)     | NaN @ **step 4105 / epoch 78** | `grads` |
| `diagnosis3.log` | `--grad_clip 1.0` (fix 1 only)          | NaN @ **step 224 / epoch 4**   | `grads` |

### 7.1 Fix 2 (`num_terms 16 -> 8`) — delays, does not prevent

- `sine_angle` dropped 102943 -> 402 (`2**7 * pi`); the sinusoidal amplifier is gone.
- Survived **6.6x longer** (step 618 -> 4105).
- But the latent kept diverging with nothing to stop it: `mu_absmax` climbed
  0.3 -> 5 -> 10 -> 15 -> **20**, `logvar` railed at the Hardtanh `+2` ceiling for
  **72 %** of entries the whole run (and the `-6` floor started railing too by
  step ~4000), β cycled to **0** every ~5 epochs, `min_kl=0`. `emb_absmax`
  reached **25**. `grad_global_norm` climbed to **1.1e17**, then `inf`.
- This time Adam's own `exp_avg_sq` for `mlp_nums.*` / `conv_layer1.weight`
  **also overflowed to Infinity** (it stayed finite in the baseline).
- Top-grad param the whole run: `encoder.Emb.mlp_output.2.weight`.

### 7.2 Fix 1 (`clip_grad_norm_(1.0)`) — makes it WORSE

Clipping alone NaN'd **3x sooner** than the baseline and spread the damage
across the **entire model** (baseline: only `encoder.Emb.*`, one row each;
run 3: every embedding, both GRUs, `fc_mu`/`fc_logvar`, the whole decoder — all
fully NaN). Two reasons, both structural:

1. **The `Inf` is created inside `backward()`** (first bad stage is `grads`,
   which is checked *before* the clip in the debug driver). `clip_grad_norm_`
   only *rescales* an existing `.grad`; it cannot un-overflow a backward pass.
2. **`clip_grad_norm_` with any non-finite grad poisons every parameter.** It
   computes `total_norm` over all grads; one `Inf` makes `total_norm = inf`, so
   `clip_coef = max_norm / (total_norm + 1e-6) -> 0`, and then every
   `p.grad.mul_(clip_coef)` does `Inf * 0 = NaN` / `finite * 0 = 0`. The next
   `optimizer.step()` writes NaN into the whole model. This is exactly the
   whole-model `[params]` NaN seen in `diagnosis3.log`.
3. `max_norm=1.0` against a raw norm of 1e5–1e9 means `clip_coef ~ 1e-5–1e-9`
   every step. Adam's second moment `v` then tracks the *clipped* (~1.0-scale)
   gradients instead of the real 1e5-scale ones, so `sqrt(v)` stays small and
   `m/sqrt(v)` (the Adam step) is *larger* in the unstable direction — the run's
   gradients were bigger earlier than the baseline's (1.7e5 vs 75 at step 150).

### 7.3 What all three runs agree on

- **First non-finite stage is always `grads`.** Forward and the loss
  (`RE`, `KL`) are finite every time. The overflow is in `backward()`.
- **`weight_global_norm` stays ~269–274** until the trip. Adam bounds the
  *weights*; weights are never the problem.
- **`grad_global_norm` explodes to 1e8–1e17** over tens–thousands of steps.
- **`logvar` railed at the Hardtanh `+2` ceiling** for 56–72 % of entries from
  step ~50, and **`mu_absmax` grows without bound** (8.5 / 20 / 6 at the three
  trips). The KL never engages (cyclic β hits 0, `min_kl=0`).
- The NaN originates in **`encoder.Emb`** (categorical embeddings + `mlp_nums`),
  and the largest gradient before the blow-up is
  **`encoder.Emb.mlp_output.2.weight`**.

**Conclusion:** this is not one bad component — it is a systemically unstable
configuration (unregularised latent + high-freq Fourier features + post-LN
transformer + `lr=4e-4`, no warmup). A rescaling clip applied *after* an
`Inf`-producing backward cannot help and actively hurts.

---

## 8. Revised solution plan

### 8.1 Must-have: skip non-finite steps (replace the naive clip)

Rescaling can't survive an `Inf`; **skipping** can. This is the `GradScaler`
pattern. In both training scripts, after `backward()`:

```python
grads = [p.grad for p in ae.parameters() if p.grad is not None]
finite = all(torch.isfinite(g).all() for g in grads)
if not finite:
    optimizer_ae.zero_grad(set_to_none=True)   # drop this batch entirely
    n_skipped += 1
    continue
total_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=10.0)
optimizer_ae.step()
```

Notes:
- `max_norm` **10–100**, not 1.0 — enough to stop fp32 overflow, loose enough
  not to fight Adam (see §7.2.3). Or clip-by-value with
  `torch.nan_to_num_(g, nan=0, posinf=1e4, neginf=-1e4)` per grad first.
- Log `n_skipped`. If more than a few per epoch, §8.2/§8.3 are not done.
- This is a guard, not a cure. It keeps a run alive so the real fixes can work.

### 8.2 Must-have: make the latent actually regularised

The latent diverges in every run. Fix all three of:

- **`fc_logvar`** — *IMPLEMENTED* (`model/timeautoencoder.py`, `Encoder`): the
  dead `Hardtanh(-6, 2)` is replaced by a plain `nn.Linear` +
  `logvar = logvar_min + (logvar_max - logvar_min) * sigmoid(raw)` in `forward`
  (same `(-6, 2)` range by default, gradient never zero). Toggle the range with
  `--logvar_min` / `--logvar_max`.
- **`mu`** — *IMPLEMENTED*: `mu = mu_clip * tanh(mu / mu_clip)` (`--mu_clip 5.0`
  default, `<=0` disables). ~identity for healthy O(1) `mu`, hard ceiling at
  `mu_clip`. NOTE: like any squash its gradient decays far out — if `mu` still
  creeps up, switch to / add the `1e-4 * mu.pow(2).mean()` loss penalty.
- **KL must never be zero-weighted** — *NOT YET DONE*: replace `frange_cycle_linear`
  with a
  *monotone* warmup to a floor that stays put, e.g. `beta = min(max_beta,
  max_beta * step / warmup_steps)` with `max_beta >= 1e-3`; and use free-bits:
  ```python
  kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B,L,lat)
  loss_kld = torch.clamp(kl_per_dim, min=free_bits).sum(-1).mean()   # free_bits ~ 0.5
  ```
  Delete the `torch.maximum(KL, delta)` / `min_kl` no-op.

### 8.3 Should-have: remove the remaining amplifiers

- **Fourier features** — *PARTLY IMPLEMENTED*: `nn.LayerNorm(emb_dim)` is applied
  to the `mlp_nums` output before the `torch.cat` into `x_emb_sum`
  (`Embedding_data`, `--fourier_layernorm` / `--no-fourier_layernorm`, on by
  default). Combine with `--sincos_num_terms 6-8`. Dropping the sine/cosine
  encoding entirely (plain `Linear -> SiLU -> Linear` on the raw numerics) is
  still the more aggressive option if this is not enough.
- **Post-LN transformer** (`get_torch_trans`, `:141`): `norm_first=True`. NOT DONE.
- **LR**: warmup 0 -> `lr` over ~500 steps, and try `lr=1e-4`. Every run is
  unstable by step ~100, i.e. from the start of real training.
- **`first_visit_scale`** (`auto_loss`, `:328`): 10.0 -> 2–3.

### 8.4 Recommended order

1. **§8.1 + §8.2** together, re-run `1_train_vae_debug.py --seed 0`. Target:
   0 skipped steps, `mu_absmax < 4`, `grad_global_norm < 1e3` steady-state.
2. If still skipping steps or `mu` still growing: add **§8.3** (transformer +
   LR warmup + Fourier LayerNorm).
3. Only once a debug run survives >200 epochs clean, port the same changes to
   `1_train_vae.py` / `1_train_vae_aclr.py`.

### 8.5 Debug-harness follow-ups

- Change `--grad_clip` in the `*_debug.py` drivers to the **skip-then-clip**
  guard of §8.1 (current behaviour rescales and is proven harmful), and add
  `--free_bits`, `--beta_warmup_steps`, `--logvar_bound {hardtanh,sigmoid}`,
  `--lr_warmup_steps` so §8.2/§8.3 can be swept from the CLI.
- Pull `metrics_tail.jsonl` / `batch.pt` from `diagnosis3.log`'s run: confirm
  the step-224 batch's raw `backward` grad is `Inf` (stage `grads`) *before*
  any clip, i.e. the overflow is independent of clipping.
