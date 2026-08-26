"""Single-file configuration for the whole MeLD pipeline.

Every entry point takes `--config <path>` and nothing else. This module loads that
file over `configs/meld_default.yaml`, resolves the derived values, and validates the
result before any stage starts work.

    from model import meld_config
    cfg = meld_config.load(path)
    kwargs = cfg.deapstack_kwargs(shapes_from_hdf5)

Two design points worth knowing:

* The default layer is `configs/meld_default.yaml`, located relative to this file, so
  there is exactly one copy of the schema and it is readable/annotated.
* `deapstack_kwargs` cross-checks itself against `DeapStack.__init__`. Adding a
  constructor argument without a config key is a startup error, which is what keeps
  "every parameter is exposed" true over time rather than just today.
"""
import difflib
import inspect
import os
import sys

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["load", "MeldConfig", "MeldConfigError"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS_PATH = os.path.join(_REPO_ROOT, "configs", "meld_default.yaml")

# Constructor arguments that come from the data, not the config.
DATA_DERIVED_KEYS = (
    "seq_len", "n_bins", "n_cats", "n_nums", "cards", "feature_size", "time_dim",
)

# create_transport() is splatted wholesale and accepts no extra keys, so an unknown
# key here is a TypeError deep inside the DiT. Mirrors transport/__init__.py, including
# the upstream `partitial_train` spelling.
TRANSPORT_KEYS = (
    "path_type", "prediction", "loss_weight", "train_eps", "sample_eps",
    "use_cosine_loss", "use_lognorm", "partitial_train", "partial_ratio", "shift_lg",
)

# Keys the vendored DiT tests with `in`, where a materialized null would change the
# branch taken (e.g. `torch.load(None)`). The adapter prunes nulls everywhere except
# inside `transport`, so these are listed for documentation and for the tests.
PRESENCE_TESTED_DIT_KEYS = ("train.weight_init", "data.image_size", "vae")


class MeldConfigError(Exception):
    """Raised for any malformed configuration, always naming the offending key."""


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _walk_keys(node, prefix=""):
    """Yield every dotted key path in a nested mapping."""
    if not isinstance(node, (dict, DictConfig)):
        return
    for key in node:
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path
        yield from _walk_keys(node[key], path)


def _reject_unknown_keys(user_cfg, defaults):
    """Fail on any key absent from the reference schema, suggesting the likely intent."""
    known = set(_walk_keys(defaults))
    # `vae.stages` and `dit.runtime.launcher` are lists; their contents are values,
    # not schema keys, so only mapping structure is compared.
    unknown = [k for k in _walk_keys(user_cfg) if k not in known]
    if not unknown:
        return
    lines = []
    for key in sorted(unknown):
        # Don't report children of an already-reported unknown parent.
        if any(key.startswith(u + ".") for u in unknown if u != key):
            continue
        match = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
        hint = f"  (did you mean '{match[0]}'?)" if match else ""
        lines.append(f"  unknown key: '{key}'{hint}")
    raise MeldConfigError(
        "configuration contains keys that are not in the schema:\n"
        + "\n".join(lines)
        + f"\n\nSee {DEFAULTS_PATH} for the full list of valid keys."
    )


def _to_plain(node):
    """Resolve interpolations and return plain Python containers."""
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


# --------------------------------------------------------------------------- #
# the config object
# --------------------------------------------------------------------------- #
class MeldConfig:
    def __init__(self, cfg, config_path):
        self._cfg = cfg
        self.config_path = os.path.abspath(config_path)

    # -- section accessors ------------------------------------------------- #
    @property
    def raw(self):
        return self._cfg

    @property
    def run(self):
        return self._cfg.run

    @property
    def vae(self):
        return self._cfg.vae

    @property
    def dit(self):
        return self._cfg.dit

    def __repr__(self):
        return f"MeldConfig({self.config_path})"

    def to_yaml(self):
        return OmegaConf.to_yaml(self._cfg, resolve=True)

    # -- model construction ------------------------------------------------ #
    def deapstack_kwargs(self, shapes):
        """Merge config knobs with data-derived shapes into DeapStack's arguments.

        `shapes` supplies DATA_DERIVED_KEYS, read from the HDF5 file. Raises if the
        union does not exactly cover the constructor signature -- that mismatch means
        a parameter has been added to the model with no way to configure it (or a
        config key no longer corresponds to anything).
        """
        from model import timeautoencoder as tae

        missing_shapes = set(DATA_DERIVED_KEYS) - set(shapes)
        if missing_shapes:
            raise MeldConfigError(
                f"missing data-derived shapes: {sorted(missing_shapes)}")

        kwargs = {k: shapes[k] for k in DATA_DERIVED_KEYS}
        kwargs["batch_size"] = int(self.vae.loader.batch_size)
        kwargs.update(_to_plain(self.vae.arch))
        kwargs.update(_to_plain(self.vae.loss))

        params = inspect.signature(tae.DeapStack.__init__).parameters
        expected = {
            n for n, p in params.items()
            if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        got = set(kwargs)
        if got != expected:
            problems = []
            for name in sorted(expected - got):
                problems.append(
                    f"  DeapStack argument '{name}' has no configuration key")
            for name in sorted(got - expected):
                problems.append(
                    f"  config key '{name}' is not a DeapStack argument")
            raise MeldConfigError(
                "configuration schema and DeapStack.__init__ have drifted apart:\n"
                + "\n".join(problems)
                + "\n\nAdd the key under vae.arch (or vae.loss) in "
                + DEFAULTS_PATH
            )
        return kwargs

    # -- derived paths ----------------------------------------------------- #
    def latent_suffix(self, checkpoint=None):
        """Suffix shared by the latent and condition filenames.

        Steps 2 and 4 build this the same way. They used to disagree -- step 2 wrote
        `latent_feature250.pt` while step 4 looked for
        `latent_feature_checkpoint_epoch_250.pt` -- so any run pinned to a checkpoint
        produced latents the decoder could not find.
        """
        if checkpoint is None:
            checkpoint = self.vae.encode.checkpoint
        if checkpoint in (None, ""):
            return ""
        return f"_checkpoint_epoch_{checkpoint}"

    def latent_path(self, checkpoint=None):
        name = str(self.vae.output.latent_template).format(
            suffix=self.latent_suffix(checkpoint))
        return os.path.join(str(self.vae.output.dir), name)

    def condition_path(self, checkpoint=None):
        name = str(self.vae.output.condition_template).format(
            suffix=self.latent_suffix(checkpoint))
        return os.path.join(str(self.vae.output.dir), name)

    def checkpoint_path(self, epoch):
        name = str(self.vae.output.checkpoint_template).format(epoch=epoch)
        return os.path.join(str(self.vae.output.dir), name)

    def best_path(self):
        return os.path.join(str(self.vae.output.dir), str(self.vae.output.best_name))

    def params_path(self):
        return os.path.join(str(self.vae.output.dir), str(self.vae.output.params_name))

    def dit_exp_dir(self):
        return os.path.join(str(self.dit.train.output_dir), str(self.dit.train.exp_name))

    def dit_ckpt_dir(self):
        return os.path.join(self.dit_exp_dir(), "checkpoints")

    def dit_sample_dir(self):
        """Reproduces the directory inference_single.py builds for its output."""
        return os.path.join(
            self.dit_exp_dir(),
            f"samples-{self.dit.sample.total}-cfg{self.dit.sample.cfg_scale}",
        )


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def _resolve_run_root(cfg, config_path):
    if cfg.run.root is not None:
        return os.path.abspath(str(cfg.run.root))
    env = os.environ.get("MELD_RUN_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.dirname(os.path.abspath(config_path))


def _abspath(root, value):
    if value is None:
        return None
    value = str(value)
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(root, value))


def _derive(cfg, config_path):
    """Fill in every `null` that has a derivation rule, then absolutize paths.

    Paths must end up absolute: the DiT reads its config with cwd set to
    model/LightningD1T/, so a surviving relative path silently resolves against the
    wrong directory.
    """
    root = _resolve_run_root(cfg, config_path)
    cfg.run.root = root
    name = str(cfg.run.name)

    # --- VAE ---
    v = cfg.vae
    if v.data.split_seed is None:
        v.data.split_seed = int(cfg.run.seed)
    if v.output.dir is None:
        v.output.dir = os.path.join(root, "save", f"vae_{name}")
    v.output.dir = _abspath(root, v.output.dir)
    v.data.h5_path = _abspath(root, v.data.h5_path)
    v.data.parser_path = _abspath(root, v.data.parser_path)

    if v.encode.h5_path is None:
        v.encode.h5_path = v.data.h5_path
    v.encode.h5_path = _abspath(root, v.encode.h5_path)
    v.encode.condition_path = _abspath(root, v.encode.condition_path)

    if v.train.beta.n_cycle is None:
        # Floor of 1: the historical int(epochs/5) was 0 for runs under 5 epochs,
        # which made the beta schedule raise ZeroDivisionError.
        v.train.beta.n_cycle = max(1, int(v.train.epochs) // 5)
    v.train.resume_from = _abspath(root, v.train.resume_from)

    if v.decode.output_dir is None:
        v.decode.output_dir = os.path.join(root, "save", "generated")
    v.decode.output_dir = _abspath(root, v.decode.output_dir)

    # --- DiT ---
    d = cfg.dit
    if d.train.output_dir is None:
        d.train.output_dir = os.path.join(root, "save", "dit")
    d.train.output_dir = _abspath(root, d.train.output_dir)
    if d.train.exp_name is None:
        d.train.exp_name = name
    if d.train.seed is None:
        d.train.seed = int(cfg.run.seed)
    d.train.weight_init = _abspath(root, d.train.weight_init)

    if d.model.in_chans is None:
        d.model.in_chans = int(v.arch.lat_dim)

    helper = MeldConfig(cfg, config_path)
    if d.data.data_path is None:
        d.data.data_path = helper.latent_path()
    d.data.data_path = _abspath(root, d.data.data_path)
    if d.data.cond_path is None:
        cond = helper.condition_path()
        d.data.cond_path = cond if os.path.exists(cond) else None
    d.data.cond_path = _abspath(root, d.data.cond_path)
    if d.sample.cond_path is None:
        d.sample.cond_path = d.data.cond_path
    d.sample.cond_path = _abspath(root, d.sample.cond_path)

    if d.runtime.clip_model_path is None:
        d.runtime.clip_model_path = os.environ.get("MELD_CLIP_PATH")
    d.runtime.clip_model_path = _abspath(root, d.runtime.clip_model_path)
    if d.runtime.materialized_config is None:
        d.runtime.materialized_config = os.path.join(
            helper.dit_exp_dir(), "_resolved_config.yaml")
    d.runtime.materialized_config = _abspath(root, d.runtime.materialized_config)
    if d.runtime.python is None:
        d.runtime.python = sys.executable

    # --- step 4, which depends on both halves ---
    if v.decode.norm_source is None:
        v.decode.norm_source = d.data.data_path
    v.decode.norm_source = _abspath(root, v.decode.norm_source)
    if v.decode.latents_path is None:
        v.decode.latents_path = os.path.join(helper.dit_sample_dir(), "samples.pt")
    v.decode.latents_path = _abspath(root, v.decode.latents_path)

    return cfg


def _validate(cfg):
    v, d = cfg.vae, cfg.dit

    valid_stages = {"train", "encode"}
    stages = set(_to_plain(v.stages))
    if not stages or not stages <= valid_stages:
        raise MeldConfigError(
            f"vae.stages must be a non-empty subset of {sorted(valid_stages)}, "
            f"got {sorted(stages)}")

    if not 0.0 < float(v.data.val_fraction) < 1.0:
        raise MeldConfigError(
            f"vae.data.val_fraction must be in (0, 1), got {v.data.val_fraction}")

    if int(v.arch.lat_dim) != int(d.model.in_chans):
        raise MeldConfigError(
            f"dit.model.in_chans ({d.model.in_chans}) must equal vae.arch.lat_dim "
            f"({v.arch.lat_dim}) -- the DiT operates on the VAE's latent channels")

    # Transformer_Block feeds `channels` in as d_model (not emb_dim), so it is
    # `channels` that must divide evenly by the head count.
    if int(v.arch.channels) > 0 and int(v.arch.channels) % int(v.arch.transformer_heads):
        raise MeldConfigError(
            f"vae.arch.channels ({v.arch.channels}) must be divisible by "
            f"transformer_heads ({v.arch.transformer_heads}) -- channels is the "
            f"transformer's d_model")

    extra = set(_to_plain(d.transport)) - set(TRANSPORT_KEYS)
    if extra:
        raise MeldConfigError(
            f"dit.transport has keys create_transport() does not accept: "
            f"{sorted(extra)}\n  valid keys: {list(TRANSPORT_KEYS)}")

    if str(d.model.model_type).endswith("/2") and int(d.model.patch_size) != 2:
        raise MeldConfigError(
            f"dit.model.model_type {d.model.model_type} implies patch_size 2, "
            f"but patch_size is {d.model.patch_size}")

    # The decoder inverts the normalization the DiT applied, using the min/max of the
    # latents the DiT trained on. LatentDataset computes those per channel but never
    # stores them, so the two paths must point at the same file.
    if not bool(v.decode.scaled) and str(v.decode.norm_source) != str(d.data.data_path):
        raise MeldConfigError(
            "vae.decode.norm_source must match dit.data.data_path (or set "
            "vae.decode.scaled: true).\n"
            f"  norm_source:        {v.decode.norm_source}\n"
            f"  dit.data.data_path: {d.data.data_path}")

    return cfg


def load(config_path, validate=True):
    """Load, merge, resolve and validate a MeLD configuration."""
    if not os.path.exists(config_path):
        raise MeldConfigError(f"config file not found: {config_path}")
    if not os.path.exists(DEFAULTS_PATH):
        raise MeldConfigError(f"reference schema missing: {DEFAULTS_PATH}")

    defaults = OmegaConf.load(DEFAULTS_PATH)
    user = OmegaConf.load(config_path)
    _reject_unknown_keys(user, defaults)

    cfg = OmegaConf.merge(defaults, user)
    cfg = _derive(cfg, config_path)
    if validate:
        _validate(cfg)
    return MeldConfig(cfg, config_path)
