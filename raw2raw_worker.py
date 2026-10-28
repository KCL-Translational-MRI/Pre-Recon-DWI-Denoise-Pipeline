# -*- coding: utf-8 -*-
"""
Worker process for Raw2Raw_tMPPCA.py (used when N_WORKERS > 1). Keep this file next to the pipeline.

Each worker loads the pipeline file once (its __main__ guard keeps the run itself from starting) and
then denoises the slices it is sent with the pipeline's own denoise_slice, so workers and the serial
path always run identical code. numpy is imported only after the BLAS thread limits are set.
"""
import importlib.util
import os
import sys
import time

_BLAS_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_pipeline = None


def init(pipeline_path, blas_threads=1):
    """Process initializer: limit BLAS threads, then load the pipeline module from its path."""
    global _pipeline
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(blas_threads)
    folder = os.path.dirname(os.path.abspath(pipeline_path))
    if folder not in sys.path:
        sys.path.insert(0, folder)                      # so denoise_recursive_tensor is found
    spec = importlib.util.spec_from_file_location("raw2raw_pipeline", pipeline_path)
    _pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pipeline)


def denoise(slice_data, cfg, W, L):
    """Denoise one slice; returns (denoised k-space, info dict, seconds)."""
    t = time.time()
    out, info = _pipeline.denoise_slice(slice_data, cfg, W, L)
    return out, info, time.time() - t
