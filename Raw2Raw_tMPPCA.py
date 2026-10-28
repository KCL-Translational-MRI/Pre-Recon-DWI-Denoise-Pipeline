# -*- coding: utf-8 -*-
"""
End-to-End TWIX Denoising Pipeline
==================================

Denoises Siemens TWIX raw data in the coil domain with (tensor) MP-PCA and writes a TWIX file
of the same structure, ready for the vendor reconstruction (e.g. the ICE simulator).

Data flow per slice:
  gather MDBs -> un-reflect -> k-space [Col, acquired Lin, Cha, Rep, TE]  (segments merged per
  readout; averages stored in the Ave counter become extra reps)
  -> IFFT(Col, Lin) -> whiten -> PE-wrap-pad -> MP-PCA (recursive tensor denoiser)
  -> crop PE -> unwhiten -> FFT(Col, Lin) -> re-reflect -> write each readout back into its own MDB

Main options (USER CONFIGURATION below)
  WINDOW_MODE     'manual' -> use `denoise_window`, e.g. (3, 3, 1).
                  'auto'   -> per slice, an odd (RO, PE, 1) window with at least WINDOW_TARGET_VOX
                              voxels (default nRep x nTE) and the most square *physical* footprint.
                              The PE/RO pixel-size ratio of the coil image built from the acquired
                              lines is (FOV_pe / FOV_ro) * nCol / (OS * R * nKeep): readout
                              oversampling doubles the FOV, not the pixel density.
  UNFOLD_MODE     'tensor' -> recursive tensor MP-PCA: one SVD per unfolding, in UNFOLD_ORDER.
                  'matrix' -> one SVD per patch: [window voxels] x [coils x reps x TEs].
  UNFOLD_ORDER    (tensor) 'size' -> spatial patch first, then the other dims by descending size.
                  or a tuple of names from "Vox", "Cha", "Rep", "TE"; names joined with "+" form one
                  unfolding, e.g. ("Vox+Cha", "Rep") = a [window voxels x coils] x [reps] matrix per
                  patch. Dims left out are appended by descending size.
  CENTER_ASSIGN   True -> each patch writes only its centre voxel (dwidenoise convention);
                  False -> overlapping patch estimates are averaged.
  USE_COL_MASK    denoise only patches inside the central nCol/OS columns (+ MASK_MARGIN_FRAC); the
                  oversampled margins, which the reconstruction crops, are written back unchanged.
                  Roughly halves the runtime.
  EXCLUDE_REPS    reps left out of denoising and written back unchanged (e.g. a partial last volume
                  of a truncated file), so they can't distort the noise / rank estimate.
  ONLY_SLICES     denoise only these slices (0-based cSlc); the others are written back unchanged.

Speed options (none changes the result beyond ~1e-6 of the noise level)
  SVD_METHOD        'gram': singular values/vectors from the small Gram matrix | 'lapack'.
  SKIP_PAD_PATCHES  with CENTER_ASSIGN, patches centred in the PE wrap padding are cropped away
                    anyway, so they are not computed.
  INDEX_FROM_HEADERS  index MDBs from header fields instead of reading every data block.
                    INDEX_SELF_CHECK also builds the data-based index and stops if they differ.
  KEEP_FILE_OPEN    one open file handle for all data reads.
  N_WORKERS         slices denoised in parallel worker processes (needs raw2raw_worker.py next to this
                    file). Run from a terminal / Anaconda Prompt; Spyder may hide worker errors.

Data layout
  Averages stored in the Ave counter (e.g. some TSE protocols) are folded into the Rep axis
  (effective rep = Rep x nAve + Ave), so every average is denoised as its own rep and written back
  to its own MDB. Each readout is placed in its own (rep, line) slot; two readouts on the same slot
  of a slice stop the run right after indexing, naming the counters that differ.
  Empty MDBs (0 channels or 0 samples, e.g. zero-filled bytes in a damaged file) are skipped.
  Noise, PHASCOR, B1 control-loop, navigator and sync data are never modified.

Requires the companion denoise_recursive_tensor.py (with mask / center_assign / svd / progress
support) and twixtools.

Directions for use:
    1. Install the required libraries and place denoise_recursive_tensor.py (and raw2raw_worker.py
       for N_WORKERS > 1) next to this file.
    2. Point DATA_ROOT / ECHO_FILES to the raw file(s). Several files (e.g. echoes) are stacked
       along an extra TE dimension and denoised jointly.
    3. Set OUTPUT_DIR (a dedicated folder is recommended: the ICE recon creates many files).
       Existing outputs are never overwritten; the run stops before processing instead.
    4. Set the options. ONLY_SLICES is handy for a quick first look at one slice.
    5. MEAS_INDEX_OVERRIDE selects the measurement (default: the one with most image blocks).
    6. Reconstruct the denoised TWIX in the ICE simulator.
"""

import os
import sys
import time
import inspect
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import scipy.fft as spfft
from twixtools import read_twix, write_twix

import denoise_recursive_tensor as _drt
from denoise_recursive_tensor import denoise_recursive_tensor


# =============================================================================
# USER CONFIGURATION
# =============================================================================
DATA_ROOT = Path(r"YOUR\PATH\HERE")
RAW_DIR = DATA_ROOT / "Raw"

# Add more files here to denoise multiple TEs jointly.
ECHO_FILES = [
    RAW_DIR / "TWIX_FILENAME.dat"
]

OUTPUT_DIR = Path(r"OUTPUT\DIRECTORY")

# ---- Window --------------------------------------------------------------
WINDOW_MODE = "manual"          # 'manual' | 'auto'
denoise_window = (3, 3, 1)      # used when WINDOW_MODE == 'manual'
WINDOW_TARGET_VOX = None        # 'auto': voxel target; None -> nRep * nTE
WINDOW_MAX = 15                 # 'auto': largest odd edge length considered
WINDOW_EXCESS_WEIGHT = 0.5      # 'auto': penalty weight on (n_vox - target) / target
PIXEL_ASPECT_OVERRIDE = None    # 'auto': PE/RO physical pixel ratio; None -> from header + data
FOV_RO_MM = None                # fallbacks if the header lookup fails (None -> assume square FOV)
FOV_PE_MM = None
OS_FACTOR_FALLBACK = 2.0        # readout oversampling if not found in the header

# ---- Denoiser ------------------------------------------------------------
CENTER_ASSIGN = True
UNFOLD_MODE = "tensor"                   # 'tensor' | 'matrix'
UNFOLD_ORDER = ("Vox+Cha", "Rep")        # 'size' | tuple of names, e.g. ("Vox+Cha", "Rep") or ("Vox", "Cha", "Rep")

# ---- Column mask ---------------------------------------------------------
USE_COL_MASK = True
MASK_MARGIN_FRAC = 0.10         # extra columns beyond nCol/OS per side, as a fraction of nCol/OS

# ---- Data ----------------------------------------------------------------
DENOISE_IMAGE = True
DENOISE_ACS = False
DO_WHITEN = True
COV_EPS = 1e-8
SEG_EPS = 1e-6
MEAS_INDEX_OVERRIDE = None      # None = auto-pick measurement with most image blocks
EXCLUDE_REPS = []               # reps not denoised, written back unchanged (e.g. [125])
ONLY_SLICES = None              # None = all | e.g. {22}: denoise only these cSlc (0-based)

# ---- Speed ---------------------------------------------------------------
N_WORKERS = 1                   # slices denoised in parallel; 1 = serial
BLAS_THREADS_PER_WORKER = 1     # linear-algebra threads per worker (keep 1 when N_WORKERS > 1)
SVD_METHOD = "gram"             # 'gram' | 'lapack'
SKIP_PAD_PATCHES = True         # skip patches centred in the PE padding (CENTER_ASSIGN only)
INDEX_FROM_HEADERS = True       # index MDBs from header fields instead of reading their data
INDEX_SELF_CHECK = True         # also build the data-based index and stop if they differ; set False
                                # once it has passed on your kind of data (it costs one extra read)
KEEP_FILE_OPEN = True           # one file handle for all data reads


# =============================================================================
# MDB CLASSIFICATION HELPERS
# =============================================================================
def _pick_main_measurement(multi_meas):
    """Pick the measurement with the most image scans."""
    best_i, best_n = 0, -1
    for i, meas in enumerate(multi_meas):
        n = sum(
            1 for mdb in meas.get("mdb", [])
            if getattr(mdb, "is_image_scan", lambda: False)()
        )
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def _try_local(mdb):
    """Convert an MDB to its local (materialized) form. Returns (ml, data) or (None, None)."""
    try:
        ml = mdb.convert_to_local()
        data = getattr(ml, "data", None)
        if data is None or getattr(data, "ndim", 0) != 2:
            return None, None
        return ml, data
    except Exception:
        return None, None


def _is_empty(mdb):
    """True for MDBs carrying no samples (0 channels or 0 samples), e.g. zero-filled bytes that
    twixtools parses as flagless headers in a damaged file."""
    try:
        return int(mdb.mdh.UsedChannels) == 0 or int(mdb.mdh.SamplesInScan) == 0
    except Exception:
        return False


def _is_flag(mdb, name):
    try:
        return bool(mdb.is_flag_set(name))
    except Exception:
        return False


def _is_noise(mdb):
    try:
        if hasattr(mdb, "is_noise_scan") and mdb.is_noise_scan():
            return True
    except Exception:
        pass
    return any(_is_flag(mdb, f) for f in ("NOISEADJSCAN", "NOISESCAN"))


def _is_phasecor(mdb):
    return _is_flag(mdb, "PHASCOR")


def _is_refscan(mdb):
    return any(_is_flag(mdb, f) for f in ("PATREFSCAN", "PATREFANDIMASCAN", "REFSCAN", "SLICE_ACCEL_REFSCAN"))


def _is_reflect(mdb):
    return _is_flag(mdb, "REFLECT")


def _is_b1_controlloop(mdb):
    return _is_flag(mdb, "B1_CONTROLLOOP")


# =============================================================================
# MATH HELPERS
# =============================================================================
def compute_whitening(noise_matrix, cov_eps=1e-8):
    """Compute whitening matrix W and its inverse L (Cholesky factor) from noise samples.

    noise_matrix: [Cha, Nsamples] complex.
    Returns (W, L) both complex64, or (None, None) if no samples available.
    """
    if noise_matrix.size == 0 or noise_matrix.shape[1] < 2:
        print("  No noise samples available; skipping whitening.")
        return None, None

    nCha = noise_matrix.shape[0]
    X = noise_matrix - np.mean(noise_matrix, axis=1, keepdims=True)
    Sigma = (X @ X.conj().T) / max(X.shape[1] - 1, 1)
    Sigma = 0.5 * (Sigma + Sigma.conj().T)
    Sigma += cov_eps * (np.trace(Sigma).real / nCha) * np.eye(nCha)
    L = np.linalg.cholesky(Sigma)
    W = np.linalg.inv(L)
    return W.astype(np.complex64), L.astype(np.complex64)


def apply_whitening(X, W, cha_axis):
    """Apply X @ W.T along the given coil axis (non-conjugate transpose)."""
    Xm = np.moveaxis(X, cha_axis, -1)
    return np.moveaxis(Xm @ W.T, -1, cha_axis)


def apply_unwhitening(X, L, cha_axis):
    """Apply X @ L.T along the given coil axis."""
    Xm = np.moveaxis(X, cha_axis, -1)
    return np.moveaxis(Xm @ L.T, -1, cha_axis)


def fft2c(x, ax1=0, ax2=1):
    """Centered FFT along two axes."""
    y = spfft.ifftshift(spfft.fft(spfft.fftshift(x, axes=[ax1]), axis=ax1, norm="backward"), axes=[ax1])
    return spfft.ifftshift(spfft.fft(spfft.fftshift(y, axes=[ax2]), axis=ax2, norm="backward"), axes=[ax2])


def ifft2c(x, ax1=0, ax2=1):
    """Centered IFFT along two axes."""
    y = spfft.ifftshift(spfft.ifft(spfft.fftshift(x, axes=[ax1]), axis=ax1, norm="backward"), axes=[ax1])
    return spfft.ifftshift(spfft.ifft(spfft.fftshift(y, axes=[ax2]), axis=ax2, norm="backward"), axes=[ax2])


def calculate_dynamic_window(n_samples):
    """Odd square window whose area is closest to n_samples (used for ACS)."""
    if n_samples < 9:
        return (3, 3, 1)
    closest = int(np.round(np.sqrt(n_samples)))
    if closest % 2 == 0:
        diff_below = abs((closest - 1) ** 2 - n_samples)
        diff_above = abs((closest + 1) ** 2 - n_samples)
        closest = (closest - 1) if diff_below < diff_above else (closest + 1)
    return (max(3, closest), max(3, closest), 1)


# =============================================================================
# UNFOLDING ORDER
# =============================================================================
def auto_order_indices(shape, spatial_group=(0, 1, 2)):
    """Index groups: spatial patch first, non-spatial dims by descending size.

    Singleton non-spatial dims are excluded - they don't contribute to the SVD and
    would trigger the `min(dims)==1` short-circuit inside the denoiser.

    Examples:
        shape=(200,84,1,32,86)    -> [[0,1,2],[4],[3]]         vox->Rep(86)->Cha(32)
        shape=(200,84,1,32,86,3)  -> [[0,1,2],[4],[3],[5]]     vox->Rep->Cha->TE
        shape=(200,36,1,52,1,1)   -> [[0,1,2],[3]]             vox->Cha only
    """
    spatial = list(spatial_group)
    non_spatial = sorted(
        [i for i in range(len(shape)) if i not in spatial and shape[i] > 1],
        key=lambda i: -shape[i]
    )
    return [spatial] + [[i] for i in non_spatial]


def matrix_order_indices(shape, spatial_group=(0, 1, 2)):
    """Matrix MP-PCA: spatial patch vs. all non-singleton non-spatial dims merged."""
    spatial = list(spatial_group)
    non_spatial = [i for i in range(len(shape)) if i not in spatial and shape[i] > 1]
    return [spatial] + ([non_spatial] if non_spatial else [])


_ORDER_AXES = {"Cha": 3, "Rep": 4, "TE": 5}       # axes of the full image shape [Col, Keep, 1, Cha, Rep, TE]


def _order_groups(order):
    """UNFOLD_ORDER tuple -> list of groups of names; "Vox+Cha" or ("Vox", "Cha") = one group."""
    groups = []
    for item in order:
        if isinstance(item, str):
            groups.append([n.strip() for n in item.split("+")])
        else:
            groups.append([str(n).strip() for n in item])
    return groups


def custom_order_indices(full_shape, squeeze_axes, order):
    """Index groups from a tuple of names ("Vox", "Cha", "Rep", "TE"). Names of singleton dims are
    skipped; non-singleton dims not named are appended by descending size. A group can merge several
    dims into one unfolding ("Vox+Cha"). Indices refer to the squeezed array."""
    kept_axes = [ax for ax in range(len(full_shape)) if ax not in squeeze_axes]
    new_index = {old: new for new, old in enumerate(kept_axes)}
    nonsp = [ax for ax in kept_axes if ax >= 3 and full_shape[ax] > 1]
    by_size = sorted(nonsp, key=lambda ax: -full_shape[ax])
    groups = _order_groups(order)
    names = [n for g in groups for n in g]
    unknown = [n for n in names if n != "Vox" and n not in _ORDER_AXES]
    if unknown or "Vox" not in names or len(set(names)) != len(names):
        raise ValueError(f"UNFOLD_ORDER {order!r}: names must be distinct, from Vox/Cha/Rep/TE, "
                         f"and include 'Vox'")
    out, used = [], set()
    for g in groups:
        idx = []
        for n in g:
            if n == "Vox":
                idx += [0, 1, 2]
            elif _ORDER_AXES[n] in nonsp:
                idx.append(new_index[_ORDER_AXES[n]])
                used.add(_ORDER_AXES[n])
        if idx:
            out.append(idx)
    out += [[new_index[ax]] for ax in by_size if ax not in used]
    return out


def select_indices(full_shape, squeeze_axes, patch_shape, unfold_mode, unfold_order):
    """Index groups for the denoiser (one place for denoise_slice and the start-up preview)."""
    if unfold_mode == "matrix":
        return matrix_order_indices(patch_shape, spatial_group=(0, 1, 2))
    if unfold_order == "size":
        return auto_order_indices(patch_shape, spatial_group=(0, 1, 2))
    return custom_order_indices(full_shape, squeeze_axes, unfold_order)


def order_tag(unfold_mode, unfold_order):
    """Output-name piece for the unfolding ('' for 'size')."""
    if unfold_mode == "matrix":
        return "_matrix"
    if unfold_order == "size":
        return ""
    letter = {"Vox": "V", "Cha": "C", "Rep": "R", "TE": "T"}
    return "_order" + "-".join("".join(letter[n] for n in g) for g in _order_groups(unfold_order))


def _describe_indices(indices, shape, axis_names, window=None):
    """Format an index list like 'Vox(9) -> Cha(30) -> Rep(21)' or 'Vox x Cha(270) -> Rep(21)'."""
    parts = []
    for g in indices:
        if len(g) > 1 and set(g) & {0, 1, 2} and not set(g) <= {0, 1, 2}:      # merged with Vox
            others = [i for i in g if i not in (0, 1, 2)]
            names = " x ".join(["Vox"] + [axis_names.get(i, f"ax{i}") for i in others])
            n_other = int(np.prod([shape[i] for i in others]))
            if isinstance(window, str):
                parts.append(f"{names}({window} x {n_other})")
            else:
                vox = int(np.prod(window)) if window is not None else int(np.prod([shape[i] for i in (0, 1, 2)]))
                parts.append(f"{names}({vox * n_other})")
        elif len(g) > 1 and set(g) <= {0, 1, 2}:
            if isinstance(window, str):
                parts.append(f"Vox({window})")
            else:
                vox = int(np.prod(window)) if window is not None else int(np.prod([shape[i] for i in g]))
                parts.append(f"Vox({vox})")
        elif len(g) > 1:
            names = " x ".join(axis_names.get(i, f"ax{i}") for i in g)
            parts.append(f"{names}({int(np.prod([shape[i] for i in g]))})")
        else:
            i = g[0]
            parts.append(f"{axis_names.get(i, f'ax{i}')}({shape[i]})")
    return " -> ".join(parts)


# =============================================================================
# GEOMETRY / WINDOW / COLUMN MASK
# =============================================================================
def _hdr_get(hdr, lookups):
    """Try a list of (section, key-path) lookups in a twixtools hdr dict; return first hit as float."""
    for section, path in lookups:
        try:
            node = hdr[section]
            for p in path:
                node = node[p]
            if node is not None:
                return float(node)
        except Exception:
            continue
    return None


def read_geometry(meas):
    """Readout/phase FOV [mm] and readout oversampling factor from the twix header, with the
    config fallbacks; prints what it used so a wrong header key is visible."""
    hdr = meas.get("hdr", {}) if isinstance(meas, dict) else {}
    fov_ro = _hdr_get(hdr, [
        ("Meas", ["ReadFoV"]),
        ("MeasYaps", ["sSliceArray", "asSlice", 0, "dReadoutFOV"]),
        ("MeasYaps", ["sSliceArray.asSlice[0].dReadoutFOV"]),
        ("Phoenix", ["sSliceArray", "asSlice", 0, "dReadoutFOV"]),
    ])
    fov_pe = _hdr_get(hdr, [
        ("Meas", ["PhaseFoV"]),
        ("MeasYaps", ["sSliceArray", "asSlice", 0, "dPhaseFOV"]),
        ("MeasYaps", ["sSliceArray.asSlice[0].dPhaseFOV"]),
        ("Phoenix", ["sSliceArray", "asSlice", 0, "dPhaseFOV"]),
    ])
    os_f = _hdr_get(hdr, [
        ("Meas", ["flReadoutOSFactor"]),
        ("MeasYaps", ["sKSpace", "flReadoutOSFactor"]),
        ("Phoenix", ["sKSpace", "flReadoutOSFactor"]),
    ])
    src = []
    if fov_ro is None:
        fov_ro, s = FOV_RO_MM, "config"
    else:
        s = "header"
    src.append(f"FOV_ro={fov_ro} ({s})")
    if fov_pe is None:
        fov_pe, s = FOV_PE_MM, "config"
    else:
        s = "header"
    src.append(f"FOV_pe={fov_pe} ({s})")
    if os_f is None or os_f <= 0:
        os_f, s = float(OS_FACTOR_FALLBACK), "fallback"
    else:
        s = "header"
    src.append(f"OS={os_f:g} ({s})")
    if fov_ro is None or fov_pe is None:
        src.append("FOV unknown -> assuming square FOV")
    print("  Geometry: " + ", ".join(src))
    return dict(fov_ro=fov_ro, fov_pe=fov_pe, os=float(os_f))


def estimate_accel(keep_lin):
    """PE acceleration = modal spacing of the acquired line indices (2 for R=2, 1 for R=1)."""
    idx = np.where(keep_lin)[0]
    if len(idx) < 2:
        return 1
    d = np.diff(idx)
    return int(max(1, np.bincount(d).argmax()))


def pixel_aspect(geom, nCol, nKeep, accel):
    """PE/RO physical pixel-size ratio of the coil image built from the acquired lines.

    RO pixel = OS*FOV_ro/nCol, PE pixel = FOV_pe/(R*nKeep)  ->  ratio = (FOV_pe/FOV_ro)*nCol/(OS*R*nKeep)
    """
    if PIXEL_ASPECT_OVERRIDE is not None:
        return float(PIXEL_ASPECT_OVERRIDE)
    fov_ratio = 1.0
    if geom.get("fov_ro") and geom.get("fov_pe"):
        fov_ratio = float(geom["fov_pe"]) / float(geom["fov_ro"])
    return fov_ratio * nCol / (geom["os"] * accel * max(nKeep, 1))


def choose_window(target_vox, pix_aspect, wmax=15, excess_weight=0.5, nCol=None, nKeep=None):
    """Odd (w_ro, w_pe, 1) with w_ro*w_pe >= target_vox and the most square physical footprint.

    Score = |log(RO extent / PE extent)| + excess_weight * (n_vox - target)/target.
    """
    target_vox = max(int(target_vox), 9)
    ro_max = min(wmax, nCol) if nCol else wmax
    pe_max = min(wmax, nKeep) if nKeep else wmax
    ro_max -= (ro_max + 1) % 2   # force odd
    pe_max -= (pe_max + 1) % 2
    best = None
    for w_ro in range(3, ro_max + 1, 2):
        for w_pe in range(3, pe_max + 1, 2):
            n = w_ro * w_pe
            if n < target_vox:
                continue
            ratio = w_ro / (w_pe * pix_aspect)          # physical RO extent / PE extent
            score = abs(np.log(ratio)) + excess_weight * (n - target_vox) / target_vox
            if best is None or score < best[0]:
                best = (score, w_ro, w_pe)
    if best is None:
        warnings.warn(f"choose_window: no odd window <= {wmax} reaches {target_vox} voxels; "
                      f"using ({ro_max},{pe_max},1).")
        return (max(3, ro_max), max(3, pe_max), 1)
    return (best[1], best[2], 1)


def col_mask_bounds(nCol, os_factor, margin_frac, center_col):
    """[c0, c1) column range covering the nominal RO FOV (nCol/OS) plus a margin per side."""
    half = 0.5 * (nCol / os_factor) * (1.0 + margin_frac)
    c0 = max(0, int(np.floor(center_col - half)))
    c1 = min(nCol, int(np.ceil(center_col + half)))
    return c0, c1


# =============================================================================
# INDEXING PASS
# =============================================================================
def _mdb_shape(mdb, from_headers, file_size=None):
    """(nCha, nCol) of an MDB's data block, or None if it holds no readable 2-D data.

    from_headers=True takes UsedChannels / SamplesInScan from the header without touching the
    disk; False reads the data block. They agree for every intact MDB; blocks running past the end
    of a truncated file are rejected in both modes.
    """
    if not from_headers:
        ml, data = _try_local(mdb)
        return None if ml is None else (int(data.shape[0]), int(data.shape[1]))
    if _is_flag(mdb, "SYNCDATA") or _is_flag(mdb, "ACQEND"):
        return None
    try:
        shape = (int(mdb.mdh.UsedChannels), int(mdb.mdh.SamplesInScan))
        pos = getattr(mdb, "mem_pos", None)
        end = int(pos) + int(mdb.dma_len) if pos is not None else None
    except Exception:
        return None
    if file_size is not None and end is not None and end > file_size:
        return None
    return shape


def _fold_averages(raw_by_slice):
    """Fold the Ave counter into Rep: effective rep = Rep * nAve + (rank of Ave), so a file with one
    average is unchanged. Returns ({slc: [(rep, seg, lin, list_idx, reflect)]}, nAve, max rep)."""
    aves = sorted({e[1] for es in raw_by_slice.values() for e in es})
    rank = {a: i for i, a in enumerate(aves)}
    n_ave = max(len(aves), 1)
    out = defaultdict(list)
    mx_rep = -1
    for slc, es in raw_by_slice.items():
        for rep, ave, seg, lin, li, refl in es:
            r = rep * n_ave + rank[ave]
            out[slc].append((r, seg, lin, li, refl))
            mx_rep = max(mx_rep, r)
    return out, n_ave, mx_rep


def _duplicate_slots(by_slice):
    """{slc: [(rep, lin, [list_idx, ...]), ...]} for (rep, line) slots holding more than one readout."""
    dups = {}
    for slc, es in by_slice.items():
        groups = defaultdict(list)
        for rep, seg, lin, li, refl in es:
            groups[(rep, lin)].append(li)
        d = [(r, l, lis) for (r, l), lis in groups.items() if len(lis) > 1]
        if d:
            dups[slc] = d
    return dups


def _classify_mdbs(mdb_list, from_headers, file_size, label):
    """Index pass over one measurement: expected_col, per-slice image/refscan entries, dims."""
    # ---- First sub-pass: detect expected_col from image scans ----
    col_counts = Counter()
    n_empty = 0
    for mdb in mdb_list:
        if getattr(mdb, "is_image_scan", lambda: False)():
            if _is_empty(mdb):
                n_empty += 1
                continue
            shp = _mdb_shape(mdb, from_headers, file_size)
            if shp is not None:
                col_counts[shp[1]] += 1
    if not col_counts:
        raise RuntimeError(f"No image scans found in {label}")
    expected_col = col_counts.most_common(1)[0][0]

    # ---- Second sub-pass: classify and record metadata (Ave folded into Rep afterwards) ----
    image_raw = defaultdict(list)
    refscan_raw = defaultdict(list)
    mx_slc_i = mx_seg_i = mx_lin_i = -1
    mx_slc_a = mx_seg_a = mx_lin_a = -1
    nCha = None

    for list_idx, mdb in enumerate(mdb_list):
        if _is_noise(mdb) or _is_phasecor(mdb) or _is_b1_controlloop(mdb) or _is_empty(mdb):
            continue
        shp = _mdb_shape(mdb, from_headers, file_size)
        if shp is None or int(shp[1]) != int(expected_col):
            continue

        slc, lin, seg, rep = int(mdb.cSlc), int(mdb.cLin), int(mdb.cSeg), int(mdb.cRep)
        ave = int(getattr(mdb, "cAve", 0))
        reflect = _is_reflect(mdb)
        if nCha is None:
            nCha = int(shp[0])

        is_slice_accel_ref = _is_flag(mdb, "SLICE_ACCEL_REFSCAN")
        if getattr(mdb, "is_image_scan", lambda: False)() and not is_slice_accel_ref:
            image_raw[slc].append((rep, ave, seg, lin, list_idx, reflect))
            mx_slc_i = max(mx_slc_i, slc)
            mx_seg_i = max(mx_seg_i, seg); mx_lin_i = max(mx_lin_i, lin)
        elif _is_refscan(mdb):
            refscan_raw[slc].append((rep, ave, seg, lin, list_idx, reflect))
            mx_slc_a = max(mx_slc_a, slc)
            mx_seg_a = max(mx_seg_a, seg); mx_lin_a = max(mx_lin_a, lin)
        else:
            active = sorted(mdb.get_active_flags()) if hasattr(mdb, "get_active_flags") else []
            warnings.warn(
                f"Unrecognized flag combination {active} (list_idx={list_idx}); "
                "defaulting to refscan classification."
            )
            refscan_raw[slc].append((rep, ave, seg, lin, list_idx, reflect))
            mx_slc_a = max(mx_slc_a, slc)
            mx_seg_a = max(mx_seg_a, seg); mx_lin_a = max(mx_lin_a, lin)

    image_by_slice, n_ave_i, mx_rep_i = _fold_averages(image_raw)
    refscan_by_slice, n_ave_a, mx_rep_a = _fold_averages(refscan_raw)

    def _dims(r, sl, sg, ln):
        return dict(nRep=r + 1 if r >= 0 else 0, nSlc=sl + 1 if sl >= 0 else 0,
                    nSeg=sg + 1 if sg >= 0 else 0, nLin=ln + 1 if ln >= 0 else 0,
                    nCha=nCha or 0, nCol=expected_col)

    return dict(expected_col=expected_col, n_empty=n_empty, nCha=nCha,
                image_by_slice=image_by_slice, refscan_by_slice=refscan_by_slice,
                image_dims=_dims(mx_rep_i, mx_slc_i, mx_seg_i, mx_lin_i),
                refscan_dims=_dims(mx_rep_a, mx_slc_a, mx_seg_a, mx_lin_a),
                n_ave_image=n_ave_i, n_ave_refscan=n_ave_a,
                dup_image=_duplicate_slots(image_by_slice),
                dup_refscan=_duplicate_slots(refscan_by_slice))


_DUP_COUNTERS = ("Seg", "Ave", "Eco", "Phs", "Set", "Par", "Ida", "Idb", "Idc", "Idd", "Ide")


def check_duplicate_slots(te_list, kinds):
    """Stop when a slice has two readouts on one (rep, line) slot: one slot holds one readout, so
    the others would be lost. Runs right after indexing, before any heavy work."""
    for t, info in enumerate(te_list):
        for kind in kinds:
            dups = info["dup_image" if kind == "image" else "dup_refscan"]
            if not dups:
                continue
            n_slots = sum(len(v) for v in dups.values())
            slc = sorted(dups)[0]
            rep, lin, lis = dups[slc][0]
            mdbs = [info["mdb_list"][li] for li in lis]
            differ = []
            for f in _DUP_COUNTERS:
                try:
                    vals = [int(getattr(m.mdh.Counter, f)) for m in mdbs]
                except Exception:
                    continue
                if len(set(vals)) > 1:
                    differ.append(f"{f} {vals}")
            what = ("differ in " + ", ".join(differ)) if differ else "have identical counters"
            raise RuntimeError(
                f"TE{t+1} {kind}: {n_slots} (rep, line) slots in {len(dups)} slices hold more than "
                f"one readout (e.g. slice {slc}, rep {rep}, line {lin}: {len(lis)} readouts that "
                f"{what}). One slot can only hold one readout, so the others would be lost. If they "
                f"are repeats that the reconstruction combines, they need their own axis.")


def build_te_index(twix_file, meas_override=None):
    """Read a Twix file and build a lightweight index of MDBs to modify.

    Returns a dict with fields:
        multi_meas       : parsed twix object (kept for write)
        meas_idx         : chosen measurement index
        mdb_list         : direct reference to meas['mdb'] (mutated at end)
        expected_col     : inferred readout length
        nCha             : number of coils
        image_by_slice   : {slc: [(rep, seg, lin, list_idx, reflect), ...]}
        refscan_by_slice : same shape as image_by_slice
        image_dims, refscan_dims : dim dicts (nRep, nSlc, nSeg, nLin, nCha, nCol)
        noise_blocks     : list of [Cha, Col] arrays (from ALL measurements in the file)
        geom             : dict(fov_ro, fov_pe, os) from the header (with fallbacks)
        fh               : shared open file handle (KEEP_FILE_OPEN) or None; closed after writing
    """
    print(f"  Reading {twix_file.name}...")
    multi_meas = read_twix(str(twix_file), keep_syncdata=True, keep_acqend=True)
    meas_idx = meas_override if meas_override is not None else _pick_main_measurement(multi_meas)
    meas = multi_meas[meas_idx]
    mdb_list = meas["mdb"]
    geom = read_geometry(meas)

    # twixtools closes the file after parsing and reopens it for every data read; give every
    # file-backed MDB one shared open handle instead.
    fh = None
    if KEEP_FILE_OPEN:
        fh = open(twix_file, "rb")
        for mm in multi_meas:
            for mdb in mm.get("mdb", []):
                if hasattr(mdb, "fid"):
                    mdb.fid = fh

    file_size = os.path.getsize(twix_file)
    idx = _classify_mdbs(mdb_list, INDEX_FROM_HEADERS, file_size, twix_file.name)
    if INDEX_SELF_CHECK:
        tc = time.time()
        other = _classify_mdbs(mdb_list, not INDEX_FROM_HEADERS, file_size, twix_file.name)
        if other != idx:
            raise RuntimeError("Index self-check FAILED: header-based and data-based indexing "
                               "disagree. Set INDEX_FROM_HEADERS = False and report this file.")
        print(f"  Index self-check passed: header- and data-based indexing agree "
              f"({time.time() - tc:.1f}s)")
    if idx["n_empty"]:
        print(f"  Skipped {idx['n_empty']} empty MDBs (0 channels or 0 samples; no data, left untouched)")

    # Noise scans may live in a separate measurement; scan ALL measurements for noise MDBs.
    noise_blocks = []
    for mm in multi_meas:
        for mdb in mm.get("mdb", []):
            if _is_noise(mdb) and not _is_empty(mdb):
                ml, data = _try_local(mdb)
                if ml is not None:
                    noise_blocks.append(data.astype(np.complex64, copy=True))
                del ml, data

    return dict(
        multi_meas=multi_meas,
        meas_idx=meas_idx,
        meas=meas,
        mdb_list=mdb_list,
        expected_col=idx["expected_col"],
        nCha=idx["nCha"],
        image_by_slice=idx["image_by_slice"],
        refscan_by_slice=idx["refscan_by_slice"],
        image_dims=idx["image_dims"],
        refscan_dims=idx["refscan_dims"],
        noise_blocks=noise_blocks,
        geom=geom,
        fh=fh,
        n_ave_image=idx["n_ave_image"],
        n_ave_refscan=idx["n_ave_refscan"],
        dup_image=idx["dup_image"],
        dup_refscan=idx["dup_refscan"],
    )


# =============================================================================
# SLICE-LEVEL DENOISING
# =============================================================================
def denoise_slice(slice_data, cfg, W, L):
    """Denoise a single slice.
    slice_data (from gather_slice):
        k     [nCol, nCand, nCha, nRep, nTE] complex64 - one column per phase-encode line that has
              a readout in this slice (segments already merged per readout)
        lines [nCand] int - the Lin index of each column (ascending)
        nLin  int - full Lin grid size (for the acceleration estimate)
    cfg: dict of options (see build_cfg()).
    Returns (denoised k, same shape as slice_data['k']; info dict).
    """
    k_c = slice_data["k"]
    lines = np.asarray(slice_data["lines"], dtype=int)
    nLin = int(slice_data["nLin"])
    nCol, nCand, nCha, nRep, nTE = k_c.shape
    info = {}

    # ---- Lines with data ----
    present = np.any(np.abs(k_c) > SEG_EPS, axis=(0, 2, 3, 4))
    keep_lin = np.zeros(nLin, dtype=bool)
    keep_lin[lines[present]] = True
    nKeep = int(np.sum(present))
    if nKeep == 0:
        return k_c.copy(), info

    # ---- Collapse to retained lines ----
    k_reduced = k_c[:, present, :, :, :]
    accel = estimate_accel(keep_lin)

    # DC column from the mean |k| profile (centre of the column mask)
    dc_col = int(np.argmax(np.abs(k_reduced).mean(axis=(1, 2, 3, 4))))

    # ---- IFFT to image domain ----
    I = ifft2c(k_reduced, ax1=0, ax2=1)
    del k_reduced

    # Add singleton slice dim so the spatial patch (Col, Lin, Sli) is 3D
    I = I[:, :, np.newaxis, :, :, :]  # [Col, Keep, 1, Cha, Rep, TE]
    if cfg["do_whiten"] and W is not None:
        I = apply_whitening(I, W, cha_axis=3)

    # ---- Window and column-mask bounds ----
    aspect = pixel_aspect(cfg["geom"], nCol, nKeep, accel)
    if cfg["window_mode"] == "auto":
        target = cfg["window_target"] if cfg["window_target"] else nRep * nTE
        window = choose_window(target, aspect, cfg["window_max"], cfg["window_excess_weight"],
                               nCol=nCol, nKeep=nKeep)
    else:
        window = tuple(int(w) for w in cfg["window"])
    if cfg["use_col_mask"]:
        c0, c1 = col_mask_bounds(nCol, cfg["geom"]["os"], cfg["mask_margin_frac"], dc_col)
    else:
        c0, c1 = 0, nCol
    info.update(nKeep=nKeep, accel=accel, pix_aspect=float(aspect), window=window, mask_cols=(c0, c1))

    # ---- PE circular pad (avoids wrap artefacts at patch edges) ----
    ext = int(np.ceil(1.5 * window[1]))
    if ext > 0:
        I = np.pad(I, ((0, 0), (ext, ext), (0, 0), (0, 0), (0, 0), (0, 0)), mode="wrap")

    # ---- Patch-centre mask (spatial dims of the padded image) ----
    mask = np.zeros(I.shape[:3], dtype=bool)
    mask[c0:c1, :, :] = True
    if cfg["skip_pad"] and cfg["center_assign"] and ext > 0:
        # With centre-assign a patch only writes its centre voxel, and everything centred in the
        # wrap padding is cropped below, so those patches are not computed (same output).
        mask[:, :ext, :] = False
        mask[:, -ext:, :] = False

    # ---- Denoise ----
    squeeze_axes = tuple(ax for ax in range(3, I.ndim) if I.shape[ax] == 1)
    patch = np.squeeze(I, axis=squeeze_axes) if squeeze_axes else I
    indices = select_indices(I.shape, squeeze_axes, patch.shape, cfg["unfold_mode"], cfg["unfold_order"])
    _names = ["Col", "Keep", "Sli", "Cha", "Rep", "TE"]
    _kept_axes = [ax for ax in range(I.ndim) if ax not in squeeze_axes]
    info["unfold"] = _describe_indices(indices, patch.shape,
                                       {new: _names[old] for new, old in enumerate(_kept_axes)},
                                       window=window)
    patch_den, sig2_map, _, _ = denoise_recursive_tensor(
        patch, window, indices=indices, mask=mask, center_assign=bool(cfg["center_assign"]),
        svd=cfg["svd_method"], progress=cfg["progress"])
    for ax in squeeze_axes:
        patch_den = np.expand_dims(patch_den, axis=ax)

    # ---- Crop PE ----
    if ext > 0:
        I_den = patch_den[:, ext:-ext, ...]
        sig2_map = sig2_map[:, ext:-ext, 0]
    else:
        I_den = patch_den
        sig2_map = sig2_map[:, :, 0]
    del patch_den, I
    region = sig2_map[c0:c1]
    info["sigma2_median"] = float(np.nanmedian(region)) if np.any(np.isfinite(region)) else float("nan")

    # ---- Unwhiten, FFT back into the compact line layout (lines without data stay zero) ----
    if cfg["do_whiten"] and L is not None:
        I_den = apply_unwhitening(I_den, L, cha_axis=3)
    I_den = np.squeeze(I_den, axis=2)
    k_den_reduced = fft2c(I_den, ax1=0, ax2=1)
    del I_den
    out = np.zeros_like(k_c)
    out[:, present] = k_den_reduced
    return out, info


def gather_slice(te_list, slc, kind, dims, rep_map=None, keep_locals=True):
    """Read one slice's MDBs into a compact k-space block.

    Returns (slice_data, slice_locals):
        slice_data   dict(k=[nCol, nCand, nCha, nRep, nTE] complex64, lines=[nCand] Lin indices,
                     nLin=full Lin grid). Every readout has its own (line, rep, TE) slot, so segments
                     are merged per readout and only lines that were read take up memory.
        slice_locals (t, list_idx, ml, rep, pos, reflect) per MDB to write back; pos indexes `lines`.
    rep_map: {original rep: compact index} of the reps to denoise; reps not in it are neither read
             nor written (EXCLUDE_REPS). None = all reps.
    With keep_locals=False the local MDB copies are not kept (ml = None) and scatter_slice
    re-creates them, so a slice in flight to a worker holds its data only once.
    """
    key = "image_by_slice" if kind == "image" else "refscan_by_slice"
    nRep = len(rep_map) if rep_map is not None else dims["nRep"]
    nSeg, nLin = dims["nSeg"], dims["nLin"]
    nCha, nCol = dims["nCha"], dims["nCol"]
    nTE = len(te_list)

    chosen = {}      # (t, rep, lin) -> (list_idx, reflect)
    for t in range(nTE):
        for rep, seg, lin, list_idx, reflect in te_list[t][key].get(slc, []):
            if rep_map is not None:
                if rep not in rep_map:
                    continue                       # excluded rep
                rep = rep_map[rep]                 # compact Rep index
            if rep >= nRep or seg >= nSeg or lin >= nLin:
                continue
            slot = (t, rep, lin)
            if slot in chosen:
                raise RuntimeError(f"slice {slc} ({kind}): two readouts on rep {rep}, line {lin} (TE{t+1})")
            chosen[slot] = (list_idx, reflect)

    lines = np.array(sorted({lin for (_, _, lin) in chosen}), dtype=int)
    pos = {int(l): i for i, l in enumerate(lines)}
    k = np.zeros((nCol, len(lines), nCha, nRep, nTE), dtype=np.complex64)
    slice_locals = []
    for (t, rep, lin), (list_idx, reflect) in chosen.items():
        ml, data = _try_local(te_list[t]["mdb_list"][list_idx])
        if ml is None:
            continue
        d = data[:, ::-1] if reflect else data          # un-reflect to logical k-space
        k[:, pos[lin], :, rep, t] = d.T.astype(np.complex64, copy=False)
        slice_locals.append((t, list_idx, ml if keep_locals else None, rep, pos[lin], reflect))
    return dict(k=k, lines=lines, nLin=nLin), slice_locals


def scatter_slice(te_list, k_den, slice_locals):
    """Write a denoised slice back into local copies of its MDBs."""
    for t, list_idx, ml, rep, pos, reflect in slice_locals:
        if ml is None:
            ml, _ = _try_local(te_list[t]["mdb_list"][list_idx])
            if ml is None:
                continue
        slab = k_den[:, pos, :, rep, t].T
        if reflect:
            slab = slab[:, ::-1]
        ml.data = slab.astype(np.complex64, copy=False)
        te_list[t]["mdb_list"][list_idx] = ml


def process_slice(te_list, slc, kind, dims, cfg, W, L, rep_map=None):
    """Gather -> denoise -> scatter a single slice (serial path). Returns (n_gathered, info)."""
    slice_data, slice_locals = gather_slice(te_list, slc, kind, dims, rep_map=rep_map)
    if not slice_locals:
        return 0, {}
    tensor_den, info = denoise_slice(slice_data, cfg, W, L)
    del slice_data
    scatter_slice(te_list, tensor_den, slice_locals)
    return len(slice_locals), info


_BLAS_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def denoise_image_parallel(te_list, slices, dims, cfg, W, L, rep_map):
    """Denoise `slices` in N_WORKERS worker processes; gather/scatter stay in this process.

    At most N_WORKERS + 1 slices are in flight, so the next slice is read while the workers
    compute. Results are written back as they finish (order does not matter).
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import raw2raw_worker
    for var in _BLAS_ENV_VARS:                       # inherited by the worker processes
        os.environ[var] = str(BLAS_THREADS_PER_WORKER)
    cfg_w = dict(cfg, progress=False)
    todo = list(slices)
    pending = {}
    n_total = len(todo)
    n_done = 0
    print(f"  {N_WORKERS} worker processes ({BLAS_THREADS_PER_WORKER} BLAS thread each); "
          f"slices finish out of order")
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=raw2raw_worker.init,
                             initargs=(str(Path(__file__).resolve()), BLAS_THREADS_PER_WORKER)) as ex:
        while todo or pending:
            while todo and len(pending) < N_WORKERS + 1:
                s = todo.pop(0)
                slice_data, locs = gather_slice(te_list, s, "image", dims, rep_map=rep_map,
                                                keep_locals=False)
                if not locs:
                    print(f"  slice {s+1:3d}/{dims['nSlc']}: no MDBs, skipped")
                    continue
                fut = ex.submit(raw2raw_worker.denoise, slice_data, cfg_w, W, L)
                pending[fut] = (s, locs)
                del slice_data
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                s, locs = pending.pop(fut)
                tensor_den, info, t_work = fut.result()
                scatter_slice(te_list, tensor_den, locs)
                del tensor_den
                n_done += 1
                print(f"  slice {s+1:3d}/{dims['nSlc']}: {len(locs):5d} MDBs denoised in "
                      f"{t_work:.1f}s (worker)  [{n_done}/{n_total} done]  [{_fmt_info(info)}]",
                      flush=True)


def _fmt_info(info):
    """One-line summary of a slice's info dict."""
    if not info:
        return ""
    parts = [f"nKeep={info.get('nKeep')}", f"R={info.get('accel')}",
             f"aspect={info.get('pix_aspect', float('nan')):.2f}",
             f"win={info.get('window')}", f"cols={info.get('mask_cols')}"]
    if info.get("unfold"):
        parts.append(info["unfold"])
    parts.append(f"sigma2(med)={info.get('sigma2_median', float('nan')):.3e}")
    return " | ".join(parts)


def build_cfg(geom):
    """Collect the options into one dict passed down to denoise_slice."""
    return dict(
        window_mode=WINDOW_MODE,
        window=denoise_window,
        window_target=WINDOW_TARGET_VOX,
        window_max=WINDOW_MAX,
        window_excess_weight=WINDOW_EXCESS_WEIGHT,
        use_col_mask=USE_COL_MASK,
        mask_margin_frac=MASK_MARGIN_FRAC,
        center_assign=CENTER_ASSIGN,
        unfold_mode=UNFOLD_MODE,
        unfold_order=UNFOLD_ORDER,
        do_whiten=DO_WHITEN,
        geom=geom,
        svd_method=SVD_METHOD,
        skip_pad=SKIP_PAD_PATCHES,
        progress=True,
    )


def check_denoiser_module():
    """Stop early if the imported denoise_recursive_tensor.py is too old for these options."""
    try:
        params = inspect.signature(denoise_recursive_tensor).parameters
    except (TypeError, ValueError):
        return
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [p for p in ("mask", "center_assign", "svd", "progress") if p not in params]
    if missing and not has_kwargs:
        raise ImportError(f"denoise_recursive_tensor.py ({getattr(_drt, '__file__', '?')}) does not accept "
                          f"{missing}; use the version shipped with this pipeline.")
    if SVD_METHOD == "gram" and not hasattr(_drt, "_svd"):
        raise ImportError("SVD_METHOD = 'gram' needs the denoise_recursive_tensor.py shipped with this "
                          "pipeline (the one with _svd); the imported copy is older.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nTE = len(ECHO_FILES)

    print("\n=== Raw Data Denoising Pipeline ===")
    print(f"Input files ({nTE} TE{'s' if nTE > 1 else ''}):")
    for f in ECHO_FILES:
        print(f"  {f.name}")
    if UNFOLD_MODE not in ("tensor", "matrix"):
        raise ValueError("UNFOLD_MODE must be 'tensor' or 'matrix'")
    if WINDOW_MODE not in ("manual", "auto"):
        raise ValueError("WINDOW_MODE must be 'manual' or 'auto'")
    print(f"Options: WINDOW_MODE={WINDOW_MODE}"
          + (f" {tuple(denoise_window)}" if WINDOW_MODE == "manual" else "")
          + f", CENTER_ASSIGN={CENTER_ASSIGN}, UNFOLD_MODE={UNFOLD_MODE}, UNFOLD_ORDER={UNFOLD_ORDER}, "
          f"USE_COL_MASK={USE_COL_MASK}, DO_WHITEN={DO_WHITEN}")
    print(f"Speed: N_WORKERS={N_WORKERS}, SVD_METHOD={SVD_METHOD}, SKIP_PAD_PATCHES={SKIP_PAD_PATCHES}, "
          f"INDEX_FROM_HEADERS={INDEX_FROM_HEADERS}, KEEP_FILE_OPEN={KEEP_FILE_OPEN}")
    if DENOISE_IMAGE or DENOISE_ACS:
        check_denoiser_module()
    if N_WORKERS > 1 and not (Path(__file__).resolve().parent / "raw2raw_worker.py").exists():
        raise FileNotFoundError("N_WORKERS > 1 needs raw2raw_worker.py next to this script.")
    timing = {}

    # Output names are fixed up front: write_twix will not overwrite, so fail before the work.
    tag = ("denoised" + order_tag(UNFOLD_MODE, UNFOLD_ORDER)) if (DENOISE_IMAGE or DENOISE_ACS) else "copy"
    out_paths = [OUTPUT_DIR / f"{Path(f).stem}_{tag}.dat" for f in ECHO_FILES]
    existing = [str(p) for p in out_paths if p.exists()]
    if existing:
        raise FileExistsError("Output already exists (write_twix will not overwrite): " + ", ".join(existing))

    # ---- [1/5] Index ----
    print("\n[1/5] Indexing Twix files...")
    t_stage = time.time()
    te_list = []
    for f in ECHO_FILES:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        te_list.append(build_te_index(f, MEAS_INDEX_OVERRIDE))
    timing["indexing"] = time.time() - t_stage

    # ---- Consistency checks ----
    expected_col = te_list[0]["expected_col"]
    img_dims = te_list[0]["image_dims"]
    acs_dims = te_list[0]["refscan_dims"]
    geom = te_list[0]["geom"]
    for t, info in enumerate(te_list):
        if info["expected_col"] != expected_col:
            raise RuntimeError(f"TE{t+1} expected_col={info['expected_col']} differs from TE1 ({expected_col})")
    for t, info in enumerate(te_list[1:], start=1):
        if info["image_dims"] != img_dims:
            raise RuntimeError(f"TE{t+1} image dims {info['image_dims']} differ from TE1 {img_dims}.\n"
                               "Joint multi-TE denoising requires matching dimensions.")
        if info["refscan_dims"] != acs_dims:
            raise RuntimeError(f"TE{t+1} refscan dims {info['refscan_dims']} differ from TE1 {acs_dims}.")
    print(f"  expected_col = {expected_col}")
    print(f"  Image dims   : {img_dims}")
    print(f"  ACS   dims   : {acs_dims}")

    for label, key in (("Image", "n_ave_image"), ("ACS", "n_ave_refscan")):
        n_ave = te_list[0][key]
        if n_ave > 1:
            print(f"  {label}: {n_ave} averages in the Ave counter -> each denoised as its own rep "
                  f"(effective rep = Rep x {n_ave} + Ave)")
    kinds = (["image"] if DENOISE_IMAGE else []) + (["refscan"] if DENOISE_ACS else [])
    check_duplicate_slots(te_list, kinds)
    print("  Readout slots: one readout per (rep, line) in every slice")

    rep_map = None
    nRep_eff = img_dims["nRep"]
    if EXCLUDE_REPS:
        excl = sorted({int(r) for r in EXCLUDE_REPS})
        unknown = [r for r in excl if not 0 <= r < img_dims["nRep"]]
        if unknown:
            raise ValueError(f"EXCLUDE_REPS {unknown} outside 0..{img_dims['nRep'] - 1}")
        keep_reps = [r for r in range(img_dims["nRep"]) if r not in excl]
        rep_map = {r: i for i, r in enumerate(keep_reps)}
        nRep_eff = len(keep_reps)
        print(f"  Excluding reps {excl} from denoising (written back unchanged); "
              f"denoising {nRep_eff}/{img_dims['nRep']} reps")

    # ---- [2/5] Whitening ----
    print("\n[2/5] Computing whitening matrices...")
    noise_blocks = te_list[0]["noise_blocks"]
    if noise_blocks:
        shapes = Counter(b.shape for b in noise_blocks)
        chas = [b.shape[0] for b in noise_blocks]
        modal_cha = max(set(chas), key=chas.count)
        kept = [b for b in noise_blocks if b.shape[0] == modal_cha]
        dropped = len(noise_blocks) - len(kept)
        print(f"  Noise block shape counts: {dict(shapes)}")
        if dropped:
            print(f"  Dropped {dropped} noise blocks with non-modal nCha.")
        noise_matrix = np.concatenate([b.reshape(modal_cha, -1) for b in kept], axis=1).astype(np.complex64)
        print(f"  Noise matrix: [Cha={modal_cha}, Nsamples={noise_matrix.shape[1]}]")
    else:
        noise_matrix = np.zeros((img_dims["nCha"], 0), dtype=np.complex64)

    W, L = (None, None)
    if DO_WHITEN:
        W, L = compute_whitening(noise_matrix, COV_EPS)
    if W is None:
        print("  NOTE: no whitening -> noise is not i.i.d. across channels.")

    n_lines_max = max((len({e[2] for e in es}) for es in te_list[0]["image_by_slice"].values()), default=0)
    per_slice_MB = nRep_eff * n_lines_max * img_dims["nCha"] * img_dims["nCol"] * nTE * 8 / (1024 ** 2)
    print(f"  Estimated per-slice k-space block: {per_slice_MB:.0f} MB "
          f"({n_lines_max} acquired lines of {img_dims['nLin']}; x ~3-5 with intermediates during "
          f"denoise, per worker)")

    # ---- [3/5] Window / ordering preview ----
    acs_non_spatial = max(acs_dims["nRep"] * nTE, img_dims["nCha"])
    cfg_img = build_cfg(geom)
    if cfg_img["window_target"] is None:
        cfg_img["window_target"] = nRep_eff * nTE
    cfg_acs = dict(cfg_img, window_target=acs_non_spatial, window=calculate_dynamic_window(acs_non_spatial))
    if WINDOW_MODE == "auto":
        print(f"  Window (IMAGE): auto, target >= {cfg_img['window_target']} voxels (chosen per slice; see log)")
        print(f"  Window (ACS)  : auto, target >= {acs_non_spatial} voxels")
    else:
        print(f"  Window (IMAGE): {cfg_img['window']}")
        print(f"  Window (ACS)  : {cfg_acs['window']}")

    axis_names = {0: "Col", 1: "Lin", 2: "Sli", 3: "Cha", 4: "Rep", 5: "TE"}

    def _preview(dims, window, label, nRep=None):
        nRep = dims["nRep"] if nRep is None else nRep
        probe = (dims["nCol"], dims["nLin"], 1, dims["nCha"], max(nRep, 1), nTE)
        squeeze_ax = tuple(ax for ax in range(3, len(probe)) if probe[ax] == 1)
        shape_eff = tuple(s for ax, s in enumerate(probe) if ax not in squeeze_ax)
        name_map = {new_ax: axis_names[old_ax] for new_ax, old_ax in
                    enumerate(ax for ax in range(len(probe)) if ax not in squeeze_ax)}
        indices = select_indices(probe, squeeze_ax, shape_eff, UNFOLD_MODE, UNFOLD_ORDER)
        win = window if WINDOW_MODE != "auto" else "auto"
        print(f"  MP-PCA ordering ({label}): {_describe_indices(indices, shape_eff, name_map, window=win)}")

    _preview(img_dims, cfg_img["window"], "IMAGE", nRep=nRep_eff)
    _preview(acs_dims, cfg_acs["window"], "ACS")

    # ---- [3/5] Image slices ----
    print(f"\n[3/5] Denoising IMAGE ({img_dims['nSlc']} slices)...")
    if ONLY_SLICES:
        bad = [s for s in ONLY_SLICES if not 0 <= int(s) < img_dims["nSlc"]]
        if bad:
            raise ValueError(f"ONLY_SLICES {sorted(bad)} outside 0..{img_dims['nSlc'] - 1}")
        img_slice_range = sorted({int(s) for s in ONLY_SLICES})
        print(f"  ONLY_SLICES set - denoising only slices {img_slice_range} (0-based cSlc); "
              "the others are written back unchanged.")
    else:
        img_slice_range = list(range(img_dims["nSlc"]))

    if DENOISE_IMAGE and img_dims["nSlc"] > 0:
        t_img = time.time()
        if N_WORKERS > 1 and len(img_slice_range) > 1:
            denoise_image_parallel(te_list, img_slice_range, img_dims, cfg_img, W, L, rep_map)
        else:
            for s in img_slice_range:
                ts = time.time()
                n, info = process_slice(te_list, s, "image", img_dims, cfg_img, W, L, rep_map=rep_map)
                print(f"  slice {s+1:3d}/{img_dims['nSlc']}: {n:5d} MDBs denoised in {time.time()-ts:.1f}s"
                      f"  [{_fmt_info(info)}]")
        timing["image denoising"] = time.time() - t_img
        print(f"  IMAGE total: {time.time()-t_img:.1f}s")
    else:
        print("  IMAGE denoising skipped.")

    # ---- [4/5] ACS ----
    print(f"\n[4/5] Denoising ACS ({acs_dims['nSlc']} slices)...")
    if DENOISE_ACS and acs_dims["nSlc"] > 0:
        t_acs = time.time()
        for s in range(acs_dims["nSlc"]):
            ts = time.time()
            n, info = process_slice(te_list, s, "refscan", acs_dims, cfg_acs, W, L)
            print(f"  slice {s+1:3d}/{acs_dims['nSlc']}: {n:5d} MDBs denoised in {time.time()-ts:.1f}s"
                  f"  [{_fmt_info(info)}]")
        timing["ACS denoising"] = time.time() - t_acs
        print(f"  ACS total: {time.time()-t_acs:.1f}s")
    else:
        print("  ACS denoising skipped.")

    # ---- [5/5] Write back ----
    print(f"\n[5/5] Writing Twix files ({tag})...")
    t_stage = time.time()
    for t, info in enumerate(te_list):
        write_twix(info["multi_meas"], str(out_paths[t]))
        n_img = sum(len(v) for v in info["image_by_slice"].values())
        n_acs = sum(len(v) for v in info["refscan_by_slice"].values())
        print(f"  TE{t+1}: {out_paths[t]}  (image={n_img}, acs={n_acs})")
        if info.get("fh") is not None:
            info["fh"].close()
    timing["writing"] = time.time() - t_stage

    total = time.time() - t0
    other = total - sum(timing.values())
    print("\nTiming: " + " | ".join(f"{k} {v:.1f}s" for k, v in timing.items()) + f" | other {other:.1f}s")
    print(f"\nPipeline complete in {total:.1f}s.")


if __name__ == "__main__":
    main()
