"""
DOE design generators — uses pyDOE3 (Python 3.12 compatible).
Each public function returns a pandas DataFrame with a leading "Run" column
followed by named factor columns at actual (un-coded) values.
"""

import numpy as np
import pandas as pd
import pyDOE3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_levels(col, low, high):
    """Map a coded column (any range) linearly to [low, high]."""
    col_min, col_max = float(col.min()), float(col.max())
    if col_min == col_max:
        return np.full(len(col), (float(low) + float(high)) / 2.0)
    return float(low) + (col - col_min) / (col_max - col_min) * (float(high) - float(low))


def _coded_to_actual(matrix, factors):
    """Convert a coded matrix (values in [-1, 0, 1]) to actual factor values."""
    names = [f["name"] for f in factors]
    df = pd.DataFrame(matrix, columns=names)
    for f in factors:
        df[f["name"]] = _encode_levels(df[f["name"]], f["low"], f["high"])
    return df


def _add_run_order(df):
    df.insert(0, "Run", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# ── 1. Two-Level Full Factorial ───────────────────────────────────────────────

def two_level_full_factorial(factors):
    """2^k full factorial (coded ±1 → actual levels)."""
    matrix = pyDOE3.ff2n(len(factors))
    return _add_run_order(_coded_to_actual(matrix, factors))


# ── 2. Fractional Factorial ───────────────────────────────────────────────────

# Standard generator strings from Box, Hunter & Hunter.
# Key: (n_factors, resolution) → generator string (independent letters + aliases)
_FRAC_FACT_GENERATORS = {
    # 3 factors
    (3, 3): "a b ab",
    # 4 factors
    (4, 3): "a b c ab",
    (4, 4): "a b c abc",
    # 5 factors
    (5, 3): "a b c ab ac",
    (5, 4): "a b c d abcd",
    (5, 5): "a b c d abcd",
    # 6 factors
    (6, 3): "a b c ab ac bc",
    (6, 4): "a b c d abc abd",
    (6, 5): "a b c d e abcde",
    # 7 factors
    (7, 3): "a b c ab ac bc abc",
    (7, 4): "a b c d abc abd acd",
    (7, 5): "a b c d e abcde abde",
    # 8 factors
    (8, 4): "a b c d e abc abd acd",
    (8, 5): "a b c d e f abcdef abcde",
    # 9 factors
    (9, 3): "a b c ab ac bc abc d ad",
    (9, 4): "a b c d e f abc abd acd",
    # 10 factors
    (10, 3): "a b c ab ac bc abc d ad bd",
    (10, 4): "a b c d e f g abcde abcdf abcg",
    # 11 factors
    (11, 3): "a b c ab ac bc abc d ad bd abd",
    # 12 factors
    (12, 3): "a b c ab ac bc abc d ad bd abd abcd",
}


def fractional_factorial(factors, resolution=3, generators=None):
    """
    Fractional factorial design.

    Parameters
    ----------
    factors    : list of factor dicts
    resolution : 3, 4, or 5
    generators : custom generator string, e.g. 'a b ab'  (overrides auto-select)
    """
    k = len(factors)
    names = [f["name"] for f in factors]

    if generators:
        gen_str = generators.strip()
    else:
        # Look up by (k, resolution); fall back to lower resolution
        gen_str = None
        for res in range(resolution, 2, -1):
            gen_str = _FRAC_FACT_GENERATORS.get((k, res))
            if gen_str:
                break
        if gen_str is None:
            raise ValueError(
                f"No standard fractional factorial generator available for "
                f"{k} factors at resolution {resolution}. "
                f"Please enter a custom generator string."
            )

    matrix = pyDOE3.fracfact(gen_str)
    n_cols = matrix.shape[1]

    if n_cols < k:
        raise ValueError(
            f"Generator '{gen_str}' only produces {n_cols} columns "
            f"but {k} factors were specified."
        )

    df = pd.DataFrame(matrix[:, :k], columns=names)
    for f in factors:
        df[f["name"]] = _encode_levels(df[f["name"]], f["low"], f["high"])

    return _add_run_order(df)


def list_fractional_factorial_options(k):
    """Return available resolutions for k factors."""
    return [res for (nf, res) in _FRAC_FACT_GENERATORS if nf == k]


# ── 3. Plackett-Burman Screening ─────────────────────────────────────────────

def plackett_burman(factors):
    """Plackett-Burman screening design."""
    k = len(factors)
    matrix = pyDOE3.pbdesign(k)[:, :k]
    df = pd.DataFrame(matrix, columns=[f["name"] for f in factors])
    for f in factors:
        df[f["name"]] = _encode_levels(df[f["name"]], f["low"], f["high"])
    return _add_run_order(df)


# ── 4. Central Composite Design ───────────────────────────────────────────────

_CCD_FACE_MAP = {
    "ccc": "circumscribed",
    "cci": "inscribed",
    "ccf": "face-centered",
}


def central_composite(factors, face="ccc", alpha="orthogonal", center=(4, 4)):
    """
    Central Composite Design.

    Parameters
    ----------
    face   : 'ccc' (circumscribed) | 'cci' (inscribed) | 'ccf' (face-centered)
    alpha  : 'orthogonal' | 'rotatable'
    center : (n_center_factorial, n_center_star)
    """
    face_str = _CCD_FACE_MAP.get(face, "circumscribed")
    matrix = pyDOE3.ccdesign(len(factors), center=center, alpha=alpha, face=face_str)
    return _add_run_order(_coded_to_actual(matrix, factors))


# ── 5. Box-Behnken Design ────────────────────────────────────────────────────

def box_behnken(factors, center=1):
    """Box-Behnken design — requires 3 or more factors."""
    if len(factors) < 3:
        raise ValueError("Box-Behnken design requires at least 3 factors.")
    matrix = pyDOE3.bbdesign(len(factors), center=center)
    return _add_run_order(_coded_to_actual(matrix, factors))


# ── 6. General Full Factorial ────────────────────────────────────────────────

def general_full_factorial(factors):
    """
    Full factorial with arbitrary number of levels per factor.
    Each factor uses its 'num_levels' field (default 3), linearly spaced
    between low and high.
    """
    level_counts = [max(2, int(f.get("num_levels", 3))) for f in factors]
    matrix = pyDOE3.fullfact(level_counts)  # coded 0 … n-1

    rows = []
    for row in matrix:
        point = {}
        for i, f in enumerate(factors):
            n = level_counts[i]
            lo, hi = float(f["low"]), float(f["high"])
            lvls = np.linspace(lo, hi, n)
            point[f["name"]] = round(lvls[int(row[i])], 6)
        rows.append(point)

    df = pd.DataFrame(rows, columns=[f["name"] for f in factors])
    return _add_run_order(df)


# ── 7. Taguchi Orthogonal Arrays ─────────────────────────────────────────────

def list_taguchi_arrays():
    """Return available Taguchi orthogonal array names."""
    return pyDOE3.list_orthogonal_arrays()


def taguchi(factors, array_name="L8(2^7)"):
    """
    Taguchi orthogonal array design.
    The first k columns of the selected array are assigned to the k factors.
    """
    k = len(factors)
    matrix = pyDOE3.get_orthogonal_array(array_name).astype(float)
    max_factors = matrix.shape[1]

    if k > max_factors:
        raise ValueError(
            f"'{array_name}' supports at most {max_factors} factors, "
            f"but {k} were provided. Choose a larger array."
        )

    matrix = matrix[:, :k]
    df = pd.DataFrame(matrix, columns=[f["name"] for f in factors])
    for f in factors:
        df[f["name"]] = _encode_levels(df[f["name"]], f["low"], f["high"])

    return _add_run_order(df)


# ── 8. Mixture Designs ───────────────────────────────────────────────────────

def simplex_lattice(factors, degree=2):
    """
    Simplex {q, m} lattice design — proportions summing to 1 scaled to
    [low, high] for each component.
    """
    from itertools import product as iproduct

    q = len(factors)
    if q < 2:
        raise ValueError("Mixture design requires at least 2 components.")

    values = [i / degree for i in range(degree + 1)]
    points = [
        combo for combo in iproduct(values, repeat=q)
        if abs(sum(combo) - 1.0) < 1e-9
    ]
    if not points:
        raise ValueError(f"No lattice points found for degree={degree}, q={q}.")

    df = pd.DataFrame(points, columns=[f["name"] for f in factors])
    for f in factors:
        lo, hi = float(f["low"]), float(f["high"])
        df[f["name"]] = df[f["name"]] * (hi - lo) + lo

    return _add_run_order(df)


def simplex_centroid(factors):
    """
    Simplex centroid design — all subsets of components blended equally,
    plus the overall centroid.
    """
    from itertools import combinations

    q = len(factors)
    if q < 2:
        raise ValueError("Mixture design requires at least 2 components.")

    points = []
    for r in range(1, q + 1):
        for combo in combinations(range(q), r):
            point = [0.0] * q
            for idx in combo:
                point[idx] = 1.0 / r
            points.append(point)

    df = pd.DataFrame(points, columns=[f["name"] for f in factors])
    for f in factors:
        lo, hi = float(f["low"]), float(f["high"])
        df[f["name"]] = df[f["name"]] * (hi - lo) + lo

    return _add_run_order(df)


# ── Design structure: replication, blocking, randomization ───────────────────

def apply_design_structure(
    df_base: pd.DataFrame,
    n_replicates: int = 1,
    n_blocks: int = 1,
    randomize: bool = True,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Apply replication, blocking, and run-order randomization to a base design.

    Columns added to the returned DataFrame (in order):
      Std Order  — sequential position in the un-randomized design
      Run Order  — actual order in which runs should be executed (after randomization)
      Block      — block number
      Replicate  — replicate number
      <factors>  — factor settings

    Blocking strategy
    -----------------
    * n_blocks == 1             → all runs in a single block
    * n_blocks == n_replicates  → each complete replicate is one block (recommended)
    * otherwise                 → runs distributed cyclically across blocks;
                                  a warning is returned as the second element of the
                                  tuple so the caller can surface it to the user.

    Randomization
    -------------
    When randomize=True, runs are shuffled independently *within* each block
    so that inter-block comparisons remain valid.

    Returns
    -------
    (DataFrame, warning_str | None)
    """
    factor_cols = [c for c in df_base.columns if c != "Run"]
    base = df_base[factor_cols].copy()
    n_base = len(base)

    # ── 1. Replicate ──────────────────────────────────────────────────────────
    frames = []
    for rep in range(1, n_replicates + 1):
        tmp = base.copy()
        tmp["Replicate"] = rep
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    n_total = len(df)

    # ── 2. Standard order (before any randomization) ──────────────────────────
    df["Std Order"] = range(1, n_total + 1)

    # ── 3. Block assignment ───────────────────────────────────────────────────
    warning = None
    if n_blocks <= 1:
        df["Block"] = 1
    elif n_blocks == n_replicates:
        # Best-practice complete block design: each replicate = one block
        df["Block"] = df["Replicate"]
    elif n_blocks < n_replicates and n_replicates % n_blocks == 0:
        # Group consecutive replicates into blocks
        reps_per_block = n_replicates // n_blocks
        df["Block"] = ((df["Replicate"] - 1) // reps_per_block) + 1
    else:
        # Fallback: distribute runs cyclically across blocks
        df["Block"] = [(i % n_blocks) + 1 for i in range(n_total)]
        if n_replicates % n_blocks != 0:
            warning = (
                f"The number of blocks ({n_blocks}) does not evenly divide the "
                f"number of replicates ({n_replicates}). Runs were distributed "
                f"cyclically across blocks. Consider using blocks = {n_replicates} "
                f"or a divisor of {n_replicates}."
            )

    # ── 4. Randomize within each block ────────────────────────────────────────
    if randomize:
        rng = np.random.default_rng(random_seed)
        block_frames = []
        for blk in sorted(df["Block"].unique()):
            blk_df = df[df["Block"] == blk].copy()
            blk_df = blk_df.sample(frac=1, random_state=int(rng.integers(0, 2**31)))
            block_frames.append(blk_df)
        df = pd.concat(block_frames, ignore_index=True)

    # ── 5. Run order (position in the final execution sequence) ───────────────
    df["Run Order"] = range(1, n_total + 1)

    # ── 6. Final column order ─────────────────────────────────────────────────
    col_order = ["Std Order", "Run Order", "Block", "Replicate"] + factor_cols
    df = df[col_order].reset_index(drop=True)

    return df, warning


# ── Public dispatcher ─────────────────────────────────────────────────────────

def generate_design(design_type: str, factors: list, options: dict = None) -> pd.DataFrame:
    """
    Generate a DOE design matrix.

    Parameters
    ----------
    design_type : str
        One of: 'two_level_full', 'fractional', 'plackett_burman',
                'ccd', 'box_behnken', 'general_factorial',
                'taguchi', 'simplex_lattice', 'simplex_centroid'
    factors : list[dict]
        Each dict must have: name, low, high.
        Optional keys: num_levels (for general_factorial).
    options : dict, optional
        Design-specific parameters:
        - fractional:       resolution (int), generators (str)
        - ccd:              face ('ccc'|'cci'|'ccf'), alpha ('orthogonal'|'rotatable'),
                            center_factorial (int), center_star (int)
        - box_behnken:      center (int)
        - taguchi:          array_name (str)
        - simplex_lattice:  degree (int)

    Returns
    -------
    pd.DataFrame  — columns: Run, <factor_name_1>, ..., <factor_name_k>
    """
    opts = options or {}

    dispatch = {
        "two_level_full":    lambda: two_level_full_factorial(factors),
        "fractional":        lambda: fractional_factorial(
                                 factors,
                                 resolution=int(opts.get("resolution", 3)),
                                 generators=opts.get("generators") or None,
                             ),
        "plackett_burman":   lambda: plackett_burman(factors),
        "ccd":               lambda: central_composite(
                                 factors,
                                 face=opts.get("face", "ccc"),
                                 alpha=opts.get("alpha", "orthogonal"),
                                 center=(
                                     int(opts.get("center_factorial", 4)),
                                     int(opts.get("center_star", 4)),
                                 ),
                             ),
        "box_behnken":       lambda: box_behnken(
                                 factors, center=int(opts.get("center", 1))
                             ),
        "general_factorial": lambda: general_full_factorial(factors),
        "taguchi":           lambda: taguchi(
                                 factors, array_name=opts.get("array_name", "L8(2^7)")
                             ),
        "simplex_lattice":   lambda: simplex_lattice(
                                 factors, degree=int(opts.get("degree", 2))
                             ),
        "simplex_centroid":  lambda: simplex_centroid(factors),
    }

    if design_type not in dispatch:
        raise ValueError(
            f"Unknown design type: '{design_type}'. "
            f"Valid options: {sorted(dispatch)}"
        )

    return dispatch[design_type]()
