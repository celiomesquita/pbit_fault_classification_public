"""Gaussian noise augmentation for minority PBIT classes.

Strategy (from ML_DEVELOPMENT_PLAN.md):
- Apply 1 % Gaussian noise only to F04, F05, and F06 in the training set.
- Noise scale = 0.01 * nanstd(signal).  Using the signal std rather than
  multiplying directly by the value avoids unstable perturbations near zero.
- Target: raise each minority class to approximately 160 training experiments.
- Augmentation is applied only to training-fold source experiments.
  Validation and test sets are never touched.

The augmentation works on the raw long/tidy signal (one row per sample) so
the same feature extraction pipeline is used for real and synthetic experiments.
The caller is responsible for running extract_features on the returned long
DataFrame and concatenating the result with the real training features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MINORITY_CLASSES: set[str] = {"F04", "F05", "F06"}
DEFAULT_TARGET   = 160
DEFAULT_NOISE    = 0.01   # fraction of signal std

# F03 is not in MINORITY_CLASSES by default (194 real samples) but can be
# included via a dict target to improve generalisation across unseen equipment.


def _build_augmented_long(
    long_df: pd.DataFrame,
    source_exp_idx: int,
    new_exp_idx: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a long-format DataFrame for one synthetic experiment."""
    src = long_df[long_df["Experiment_Index"] == source_exp_idx].sort_values("Sample_Index").copy()
    signal = src["Measurement"].values.astype(float)

    signal_std = float(np.nanstd(signal))
    # For near-zero-variance signals (e.g. dead F06 actuator), fall back to
    # a small absolute noise so augmentation still produces a distinct sample.
    scale = noise_scale * signal_std if signal_std > 1e-9 else noise_scale
    noise = rng.normal(0.0, scale, size=signal.shape)

    src["Measurement"]      = signal + noise
    src["Experiment_Index"] = new_exp_idx
    return src


def augment_training_set(
    long_df: pd.DataFrame,
    train_experiment_ids: set[int] | list[int],
    target_per_class: int | dict[str, int] = DEFAULT_TARGET,
    noise_scale: float                      = DEFAULT_NOISE,
    random_state: int                       = 42,
) -> pd.DataFrame:
    """Generate synthetic experiments for minority classes in the training set.

    Parameters
    ----------
    long_df : pd.DataFrame
        Clean long/tidy dataset (output of ``preprocess``).
    train_experiment_ids : set[int] or list[int]
        Experiment indices assigned to the training fold.
    target_per_class : int or dict[str, int]
        When an int, applies that target to all classes in ``MINORITY_CLASSES``
        (F04, F05, F06).  When a dict, the keys determine which classes to
        augment and the values set per-class targets.  Example::

            target_per_class = {"F03": 400, "F04": 160, "F05": 160, "F06": 160}

    noise_scale : float
        Noise amplitude as a fraction of the per-experiment signal std.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame containing only the *synthetic* experiments.
        Returns an empty DataFrame if no augmentation is needed.
    """
    rng = np.random.default_rng(random_state)
    train_ids = set(train_experiment_ids)

    train_long = long_df[long_df["Experiment_Index"].isin(train_ids)]
    train_meta = (
        train_long[["Experiment_Index", "Fault_Mode"]]
        .drop_duplicates("Experiment_Index")
    )

    if isinstance(target_per_class, dict):
        class_target_map: dict[str, int] = target_per_class
    else:
        class_target_map = {c: target_per_class for c in MINORITY_CLASSES}

    # Next available experiment index (synthetic ones get indices > max real)
    next_idx = int(long_df["Experiment_Index"].max()) + 1

    augmented_chunks: list[pd.DataFrame] = []

    for fault_mode in sorted(class_target_map.keys()):
        target = class_target_map[fault_mode]
        class_exps = train_meta.loc[
            train_meta["Fault_Mode"] == fault_mode, "Experiment_Index"
        ].tolist()

        n_real   = len(class_exps)
        n_needed = max(0, target - n_real)

        if n_needed == 0 or n_real == 0:
            continue

        sources = rng.choice(class_exps, size=n_needed, replace=True)
        for src_idx in sources:
            aug = _build_augmented_long(long_df, int(src_idx), next_idx,
                                        noise_scale, rng)
            augmented_chunks.append(aug)
            next_idx += 1

    if not augmented_chunks:
        return pd.DataFrame(columns=long_df.columns)

    return pd.concat(augmented_chunks, ignore_index=True)


def augmentation_report(
    train_df: pd.DataFrame,
    aug_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a before/after count table for the augmented training set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Real training feature matrix (metadata columns present).
    aug_features_df : pd.DataFrame
        Feature matrix for the *synthetic* experiments only.

    Returns
    -------
    pd.DataFrame
        Columns: Fault_Mode, Real, Synthetic, Total.
    """
    real_counts = train_df["Fault_Mode"].value_counts().rename("Real")
    aug_counts  = aug_features_df["Fault_Mode"].value_counts().rename("Synthetic") if len(aug_features_df) else pd.Series(dtype=int, name="Synthetic")

    report = pd.concat([real_counts, aug_counts], axis=1).fillna(0).astype(int)
    report.index.name = "Fault_Mode"
    report["Total"] = report["Real"] + report["Synthetic"]
    return report.reset_index().sort_values("Fault_Mode")
