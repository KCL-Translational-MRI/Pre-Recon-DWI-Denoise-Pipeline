# -*- coding: utf-8 -*-
"""
End-to-End TWIX Denoising Pipeline
=============================================

Data flow per slice:
  gather MDBs -> un-reflect -> k-space [Col,Lin,Cha,Seg,Rep,TE]
  merge segs -> Lin-collapse -> IFFT(Col,Lin) -> whiten -> PE-wrap-pad
  -> MP-PCA (shared recursive tensor denoiser) -> crop PE -> unwhiten
  -> FFT(Col,Lin) -> re-embed to full Lin -> split back to segs
  -> re-reflect -> write into local MDB copies
  
Directions for use:
    
    1. Ensure required libraries are installed and the companion
       denoise_recursive_tensor is visible.
    2. Direct DATA_ROOT to the location of the raw files to be denoised.
       Current structure assumes a folder within this destination titled "Raw".
    3. Specify file name(s) to be denoised in ECHO_FILES. Multiple files can be
       included and will be stacked along an additional dimension. Single files
       can be denoised and this process will be automatically bypassed.
    4. Specify the output directory for the denoised TWIX to be saved. It is 
       recommended for this to be a dedicated folder, as the ICE recon step
       can create an abundance of companion files. 
    5. If desired, specify a denoising window size using denoise_window. If 
       left blank, this will be automatically caclualated as the nearest odd
       integer greater than sqrt(vol). 
    6. Specify DENOISE_IMAGE as true to denoise image blocks. The option exists
       to denoise ACS lines.
    7. MEAS_INDEX_OVERRIDE can be used to specify the measurement from the TWIX
       file to denoise, but this defaults to the one containing the most
       image blocks. Typically the first measurement is a noise scan and the
       second is image data. 
    8. Reconstruct denoised TWIX file in ICE simulator within Siemens IDEA 
       environment. 
       
  
"""

import os
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import scipy.fft as spfft
from twixtools import read_twix, write_twix

from denoise_recursive_tensor import denoise_recursive_tensor


# =============================================================================
# USER CONFIGURATION
# =============================================================================
DATA_ROOT = Path(
    r"YOUR\PATH\HERE"
)
RAW_DIR = DATA_ROOT / "Raw"

# Add more files here to denoise multiple TEs jointly.
ECHO_FILES = [
    RAW_DIR / "TWIX_FILENAME.dat"
]

OUTPUT_DIR = Path(r"OUTPUT\DIRECTORY")

denoise_window = () # Specify a window size to denoise image, or leave empty to calculate automatically from data size

DENOISE_IMAGE = True
DENOISE_ACS = False    
DO_WHITEN = True
COV_EPS = 1e-8
SEG_EPS = 1e-6
MEAS_INDEX_OVERRIDE = None  # None = auto-pick measurement with most image blocks


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
    return any(_is_flag(mdb, f) for f in ("PATREFSCAN", "PATREFANDIMASCAN", "REFSCAN"))


def _is_reflect(mdb):
    return _is_flag(mdb, "REFLECT")


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
    """Apply X @ W.T along the given coil axis (NON-conjugate transpose, matches MATLAB v24)."""
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
    """Pick the odd-sized spatial window whose area is closest to n_samples."""
    if n_samples < 9:
        return (3, 3, 1)
    closest = int(np.round(np.sqrt(n_samples)))
    if closest % 2 == 0:
        diff_below = abs((closest - 1) ** 2 - n_samples)
        diff_above = abs((closest + 1) ** 2 - n_samples)
        closest = (closest - 1) if diff_below < diff_above else (closest + 1)
    return (max(3, closest), max(3, closest), 1)


def auto_order_indices(shape, spatial_group=(0, 1, 2)):
    """Build MP-PCA index groups: spatial patch first, non-spatial dims by descending size.

    The convention for recursive tensor MP-PCA:
        1. Spatial patch unfolds first (largest "sample" count per window position).
        2. Remaining feature dims unfold in descending size so each subsequent SVD
           operates on the matrix with the clearest signal/noise gap before the
           tensor gets compressed further down the chain.

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


def _describe_indices(indices, shape, axis_names, window=None):
    """Format an index list as a human-readable string like 'Vox(81) -> Rep(86) -> Cha(32)'.

    If `window` is given, the spatial (multi-axis) group is labelled with the product
    of the window sizes (actual per-patch voxel count) instead of the tensor-level
    spatial product.
    """
    parts = []
    for g in indices:
        if len(g) > 1:
            vox = int(np.prod(window)) if window is not None else int(np.prod([shape[i] for i in g]))
            parts.append(f"Vox({vox})")
        else:
            i = g[0]
            parts.append(f"{axis_names.get(i, f'ax{i}')}({shape[i]})")
    return " -> ".join(parts)


# =============================================================================
# INDEXING PASS
# =============================================================================
def build_te_index(twix_file, meas_override=None):
    """Read a Twix file and build a lightweight index of MDBs to modify.

    Does NOT retain local MDB copies - it only records the metadata we need
    to locate and modify each image/refscan MDB later. Data is pulled in
    just-in-time during the slice loop.

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
    """
    print(f"  Reading {twix_file.name}...")
    multi_meas = read_twix(str(twix_file), keep_syncdata=True, keep_acqend=True)
    meas_idx = meas_override if meas_override is not None else _pick_main_measurement(multi_meas)
    meas = multi_meas[meas_idx]
    mdb_list = meas["mdb"]

    # ---- First sub-pass: detect expected_col from image scans ----
    col_counts = Counter()
    for mdb in mdb_list:
        if getattr(mdb, "is_image_scan", lambda: False)():
            ml, data = _try_local(mdb)
            if ml is not None:
                col_counts[data.shape[1]] += 1
            del ml, data
    if not col_counts:
        raise RuntimeError(f"No image scans found in {twix_file.name}")
    expected_col = col_counts.most_common(1)[0][0]

    # ---- Second sub-pass: classify and record metadata ----
    image_by_slice = defaultdict(list)
    refscan_by_slice = defaultdict(list)
    noise_blocks = []

    # Noise scans may live in a separate measurement (e.g. Siemens often puts
    # pre-scan/noise-adjust in scan 0 and imaging in scan 1). Scan ALL measurements
    # for noise MDBs, not just the selected one.
    for mm in multi_meas:
        for mdb in mm.get("mdb", []):
            if _is_noise(mdb):
                ml, data = _try_local(mdb)
                if ml is not None:
                    noise_blocks.append(data.astype(np.complex64, copy=True))
                del ml, data

    mx_rep_i = mx_slc_i = mx_seg_i = mx_lin_i = -1
    mx_rep_a = mx_slc_a = mx_seg_a = mx_lin_a = -1
    nCha = None

    for list_idx, mdb in enumerate(mdb_list):
        # Noise already handled above; skip here (don't re-index into image/refscan).
        if _is_noise(mdb):
            continue

        # Phasecor: pass through, don't index (we won't touch it).
        if _is_phasecor(mdb):
            continue

        ml, data = _try_local(mdb)
        if ml is None:
            continue

        # RTFeedback and other non-EPI blocks: skip (not our readout length)
        if int(data.shape[1]) != int(expected_col):
            del ml, data
            continue

        slc = int(getattr(ml, "cSlc", 0))
        lin = int(getattr(ml, "cLin", 0))
        seg = int(getattr(ml, "cSeg", 0))
        rep = int(getattr(ml, "cRep", 0))
        reflect = _is_reflect(mdb)

        if nCha is None:
            nCha = int(data.shape[0])

        if getattr(mdb, "is_image_scan", lambda: False)():
            image_by_slice[slc].append((rep, seg, lin, list_idx, reflect))
            mx_rep_i = max(mx_rep_i, rep); mx_slc_i = max(mx_slc_i, slc)
            mx_seg_i = max(mx_seg_i, seg); mx_lin_i = max(mx_lin_i, lin)
        elif _is_refscan(mdb):
            refscan_by_slice[slc].append((rep, seg, lin, list_idx, reflect))
            mx_rep_a = max(mx_rep_a, rep); mx_slc_a = max(mx_slc_a, slc)
            mx_seg_a = max(mx_seg_a, seg); mx_lin_a = max(mx_lin_a, lin)
        else:
            # Other flag combinations - treat as refscan (matches v1)
            refscan_by_slice[slc].append((rep, seg, lin, list_idx, reflect))
            mx_rep_a = max(mx_rep_a, rep); mx_slc_a = max(mx_slc_a, slc)
            mx_seg_a = max(mx_seg_a, seg); mx_lin_a = max(mx_lin_a, lin)

        # Release the local copy immediately; we'll re-fetch during the slice loop.
        del ml, data

    image_dims = dict(
        nRep=mx_rep_i + 1 if mx_rep_i >= 0 else 0,
        nSlc=mx_slc_i + 1 if mx_slc_i >= 0 else 0,
        nSeg=mx_seg_i + 1 if mx_seg_i >= 0 else 0,
        nLin=mx_lin_i + 1 if mx_lin_i >= 0 else 0,
        nCha=nCha or 0,
        nCol=expected_col,
    )
    refscan_dims = dict(
        nRep=mx_rep_a + 1 if mx_rep_a >= 0 else 0,
        nSlc=mx_slc_a + 1 if mx_slc_a >= 0 else 0,
        nSeg=mx_seg_a + 1 if mx_seg_a >= 0 else 0,
        nLin=mx_lin_a + 1 if mx_lin_a >= 0 else 0,
        nCha=nCha or 0,
        nCol=expected_col,
    )

    return dict(
        multi_meas=multi_meas,
        meas_idx=meas_idx,
        meas=meas,
        mdb_list=mdb_list,
        expected_col=expected_col,
        nCha=nCha,
        image_by_slice=image_by_slice,
        refscan_by_slice=refscan_by_slice,
        image_dims=image_dims,
        refscan_dims=refscan_dims,
        noise_blocks=noise_blocks,
    )


# =============================================================================
# SLICE-LEVEL DENOISING
# =============================================================================
def denoise_slice(slice_tensor, window, W, L, do_whiten):
    """Denoise a single slice.

    slice_tensor: [nCol, nLin, nCha, nSeg, nRep, nTE] complex64
    Returns an array of the same shape with denoised k-space.
    """
    nCol, nLin, nCha, nSeg, nRep, nTE = slice_tensor.shape

    # ---- Segment-level line presence mask ----
    linmask = np.zeros((nLin, nSeg, nTE), dtype=bool)
    for t in range(nTE):
        for g in range(nSeg):
            linmask[:, g, t] = np.any(np.abs(slice_tensor[:, :, :, g, :, t]) > SEG_EPS, axis=(0, 2, 3))
    keep_lin = np.any(linmask, axis=(1, 2))
    nKeep = int(np.sum(keep_lin))
    if nKeep == 0:
        return slice_tensor.copy()

    # ---- Merge segments into a single k-space (first seg wins at each line) ----
    k_merged = np.zeros((nCol, nLin, nCha, nRep, nTE), dtype=np.complex64)
    filled = np.zeros((nLin, nTE), dtype=bool)
    for t in range(nTE):
        for g in range(nSeg):
            idx = np.where(linmask[:, g, t])[0]
            write_idx = idx[~filled[idx, t]]
            if len(write_idx):
                k_merged[:, write_idx, :, :, t] = slice_tensor[:, write_idx, :, g, :, t]
                filled[write_idx, t] = True

    # ---- Collapse to retained lines, IFFT to image domain ----
    k_reduced = k_merged[:, keep_lin, :, :, :]
    del k_merged
    I = ifft2c(k_reduced, ax1=0, ax2=1)
    del k_reduced

    # Add singleton slice dim so the spatial patch (Col,Lin,Sli) is 3D
    I = I[:, :, np.newaxis, :, :, :]  # [Col, Keep, 1, Cha, Rep, TE]

    if do_whiten and W is not None:
        I = apply_whitening(I, W, cha_axis=3)

    # ---- PE circular pad (avoids wrap artefacts at patch edges) ----
    ext = int(np.ceil(1.5 * window[1]))
    if ext > 0:
        I = np.pad(I, ((0, 0), (ext, ext), (0, 0), (0, 0), (0, 0), (0, 0)), mode="wrap")

    # ---- Denoise ----
    # Squeeze out any non-spatial dims that are singletons (e.g. TE=1 for single-TE
    # runs, or Rep=1 for ACS). Keep axes 0,1,2 (spatial patch) intact.
    squeeze_axes = tuple(ax for ax in range(3, I.ndim) if I.shape[ax] == 1)
    patch = np.squeeze(I, axis=squeeze_axes) if squeeze_axes else I

    # Order index groups: spatial patch first, non-spatial by descending size.
    indices = auto_order_indices(patch.shape, spatial_group=(0, 1, 2))

    patch_den, _sig2, _P, _snr = denoise_recursive_tensor(patch, window, indices=indices)

    # Restore squeezed singleton dims so downstream shape arithmetic still works.
    for ax in squeeze_axes:
        patch_den = np.expand_dims(patch_den, axis=ax)

    # ---- Crop PE ----
    if ext > 0:
        I_den = patch_den[:, ext:-ext, ...]
    else:
        I_den = patch_den
    del patch_den, I

    if do_whiten and L is not None:
        I_den = apply_unwhitening(I_den, L, cha_axis=3)

    # Drop singleton slice dim, FFT back, re-embed to full Lin grid
    I_den = np.squeeze(I_den, axis=2)
    k_den_reduced = fft2c(I_den, ax1=0, ax2=1)
    del I_den

    k_den_full = np.zeros((nCol, nLin, nCha, nRep, nTE), dtype=np.complex64)
    k_den_full[:, keep_lin, :, :, :] = k_den_reduced
    del k_den_reduced

    # ---- Split segments back matching the original linmask placement ----
    out = np.zeros_like(slice_tensor)
    for t in range(nTE):
        for g in range(nSeg):
            idx = np.where(linmask[:, g, t])[0]
            if len(idx):
                out[:, idx, :, g, :, t] = k_den_full[:, idx, :, :, t]
    return out


def process_slice(te_list, slc, kind, dims, window, W, L, do_whiten):
    """Gather -> denoise -> scatter a single slice.

    kind: 'image' or 'refscan'.

    Memory per call is bounded by the slice tensor (nRep*nSeg*nLin*nCha*nCol*nTE
    complex64) plus a few intermediate arrays during denoise.
    """
    key = "image_by_slice" if kind == "image" else "refscan_by_slice"
    nRep, nSeg, nLin = dims["nRep"], dims["nSeg"], dims["nLin"]
    nCha, nCol = dims["nCha"], dims["nCol"]
    nTE = len(te_list)

    tensor = np.zeros((nCol, nLin, nCha, nSeg, nRep, nTE), dtype=np.complex64)
    slice_locals = []  # (te_idx, list_idx, ml, rep, seg, lin, reflect)
    n_gathered = 0

    # ---- Gather ----
    for t in range(nTE):
        entries = te_list[t][key].get(slc, [])
        for rep, seg, lin, list_idx, reflect in entries:
            mdb = te_list[t]["mdb_list"][list_idx]
            ml, data = _try_local(mdb)
            if ml is None:
                continue
            if rep >= nRep or seg >= nSeg or lin >= nLin:
                continue
            d = data[:, ::-1] if reflect else data  # un-reflect to get logical k-space
            tensor[:, lin, :, seg, rep, t] = d.T.astype(np.complex64, copy=False)
            slice_locals.append((t, list_idx, ml, rep, seg, lin, reflect))
            n_gathered += 1

    if n_gathered == 0:
        return 0

    # ---- Denoise ----
    tensor_den = denoise_slice(tensor, window, W, L, do_whiten)
    del tensor

    # ---- Scatter: write denoised k-space back into local MDB .data, swap into mdb_list ----
    for t, list_idx, ml, rep, seg, lin, reflect in slice_locals:
        slab = tensor_den[:, lin, :, seg, rep, t].T  # [Cha, Col]
        if reflect:
            slab = slab[:, ::-1]
        ml.data = slab.astype(np.complex64, copy=False)
        te_list[t]["mdb_list"][list_idx] = ml

    return n_gathered


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nTE = len(ECHO_FILES)

    print("\n=== Memory-optimized denoising pipeline v2 ===")
    print(f"Input files ({nTE} TE{'s' if nTE > 1 else ''}):")
    for f in ECHO_FILES:
        print(f"  {f.name}")

    # ---- [1/5] Index ----
    print("\n[1/5] Indexing Twix files...")
    te_list = []
    for f in ECHO_FILES:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
        te_list.append(build_te_index(f, MEAS_INDEX_OVERRIDE))

    # ---- Consistency checks ----
    expected_col = te_list[0]["expected_col"]
    img_dims = te_list[0]["image_dims"]
    acs_dims = te_list[0]["refscan_dims"]

    for t, info in enumerate(te_list):
        if info["expected_col"] != expected_col:
            raise RuntimeError(
                f"TE{t+1} expected_col={info['expected_col']} differs from TE1 ({expected_col})"
            )
    for t, info in enumerate(te_list[1:], start=1):
        if info["image_dims"] != img_dims:
            raise RuntimeError(
                f"TE{t+1} image dims {info['image_dims']} differ from TE1 {img_dims}.\n"
                "Joint multi-TE denoising requires matching dimensions."
            )
        if info["refscan_dims"] != acs_dims:
            raise RuntimeError(
                f"TE{t+1} refscan dims {info['refscan_dims']} differ from TE1 {acs_dims}."
            )

    print(f"  expected_col = {expected_col}")
    print(f"  Image dims   : {img_dims}")
    print(f"  ACS   dims   : {acs_dims}")

    # ---- [2/5] Whitening ----
    print("\n[2/5] Computing whitening matrices...")
    noise_blocks = te_list[0]["noise_blocks"]  # collected across all measurements in the file
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

    # Per-slice memory peak estimate (helps spot disasters before the slice loop starts)
    per_slice_MB = (
        img_dims["nRep"] * img_dims["nSeg"] * img_dims["nLin"]
        * img_dims["nCha"] * img_dims["nCol"] * nTE * 8 / (1024 ** 2)
    )
    print(f"  Estimated per-slice gather-tensor: {per_slice_MB:.0f} MB"
          f"  (x ~3-5 with intermediates during denoise)")

    # ---- [3/5] Window selection ----
    image_non_spatial = img_dims["nRep"] * nTE
    acs_non_spatial = max(acs_dims["nRep"] * nTE, img_dims["nCha"])  # fallback on coils if nRep=1
    window_img = denoise_window if denoise_window else calculate_dynamic_window(image_non_spatial)
    window_acs = calculate_dynamic_window(acs_non_spatial)
    print(f"  Window (IMAGE): {window_img}")
    print(f"  Window (ACS)  : {window_acs}")

    # Preview the auto-chosen MP-PCA index ordering so it's visible before the slice loop.
    # We don't know exact nKeep per slice yet, but non-spatial sizes are fixed so the
    # ordering is deterministic.
    axis_names = {0: "Col", 1: "Lin", 2: "Sli", 3: "Cha", 4: "Rep", 5: "TE"}

    def _preview(dims, nTE_eff, window, label):
        probe = (dims["nCol"], dims["nLin"], 1, dims["nCha"], max(dims["nRep"], 1), nTE_eff)
        squeeze_ax = tuple(ax for ax in range(3, len(probe)) if probe[ax] == 1)
        shape_eff = tuple(s for ax, s in enumerate(probe) if ax not in squeeze_ax)
        name_map = {new_ax: axis_names[old_ax] for new_ax, old_ax in
                    enumerate(ax for ax in range(len(probe)) if ax not in squeeze_ax)}
        indices = auto_order_indices(shape_eff)
        print(f"  MP-PCA ordering ({label}): "
              f"{_describe_indices(indices, shape_eff, name_map, window=window)}")

    _preview(img_dims, nTE, window_img, "IMAGE")
    _preview(acs_dims, nTE, window_acs, "ACS")

    # ---- [4/5] Slice loop ----
    print(f"\n[3/5] Denoising IMAGE ({img_dims['nSlc']} slices)...")
    if DENOISE_IMAGE and img_dims["nSlc"] > 0:
        t_img = time.time()
        for s in range(img_dims["nSlc"]):
            ts = time.time()
            n = process_slice(te_list, s, "image", img_dims, window_img, W, L, DO_WHITEN)
            print(f"  slice {s+1:3d}/{img_dims['nSlc']}: {n:5d} MDBs denoised in {time.time()-ts:.1f}s")
        print(f"  IMAGE total: {time.time()-t_img:.1f}s")
    else:
        print("  IMAGE denoising skipped.")

    print(f"\n[4/5] Denoising ACS ({acs_dims['nSlc']} slices)...")
    if DENOISE_ACS and acs_dims["nSlc"] > 0:
        t_acs = time.time()
        for s in range(acs_dims["nSlc"]):
            ts = time.time()
            n = process_slice(te_list, s, "refscan", acs_dims, window_acs, W, L, DO_WHITEN)
            print(f"  slice {s+1:3d}/{acs_dims['nSlc']}: {n:5d} MDBs denoised in {time.time()-ts:.1f}s")
        print(f"  ACS total: {time.time()-t_acs:.1f}s")
    else:
        print("  ACS denoising skipped.")

    # ---- [5/5] Write back ----
    print("\n[5/5] Writing denoised Twix files...")
    for t, info in enumerate(te_list):
        out_path = OUTPUT_DIR / f"{Path(ECHO_FILES[t]).stem}_denoised.dat"
        write_twix(info["multi_meas"], str(out_path))
        n_img = sum(len(v) for v in info["image_by_slice"].values())
        n_acs = sum(len(v) for v in info["refscan_by_slice"].values())
        print(f"  TE{t+1}: {out_path}  (image={n_img}, acs={n_acs})  [write skipped]")

    print(f"\nPipeline complete in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
