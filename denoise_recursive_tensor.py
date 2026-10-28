# -*- coding: utf-8 -*-
"""
Python Port of denoise_recursive_tensor.m from: 

J. L. Olesen, A. Ianus, L. Østergaard, N. Shemesh, and S. N. Jespersen, 
‘Tensor denoising of multidimensional MRI data’, Magn. Reson. Med., 
vol. 89, no. 3, 2023, doi: 10.1002/mrm.29478.


Additions (v9 pipeline), all off by default so the output is unchanged:
  num_stages   truncate only the first k SVDs of the recursion (in the index-group order);
               later SVDs keep every component, which is a lossless rotation. None = all.
  shrink_rule  'frob' (default: optimal shrinkage for Frobenius loss, as in the original) or
               'operator' (optimal for operator-norm loss: subtracts about half as much from
               each kept component, so weak, barely detected components are cut much less).
               Shrinkage is applied to the LAST truncated SVD, as before; opt_shrink=False
               switches it off.
"""

# denoise_recursive_tensor.py

import numpy as np
import sys

FEATURES = frozenset({"num_stages", "shrink_rule"})   # checked by Raw2Raw v9 at start-up


def _svd(X, method="gram"):
    """Thin SVD with the same contract as np.linalg.svd(X, full_matrices=False): U, S, Vh.

    method='gram'   eigendecomposition of the smaller Gram matrix (X X^H or X^H X) in double
                    precision. MP-PCA needs exactly these singular values and vectors; this is
                    several times faster for the short-and-wide matrices of patch denoising and
                    gives the same result.
    method='lapack' np.linalg.svd, as in the original port.
    """
    if method == "lapack":
        return np.linalg.svd(X, full_matrices=False)
    X = np.asarray(X)
    Xd = X.astype(np.complex128) if np.iscomplexobj(X) else X.astype(np.float64)
    M, N = Xd.shape
    if M <= N:
        w, U = np.linalg.eigh(Xd @ Xd.conj().T)
        w, U = w[::-1], U[:, ::-1]
        S = np.sqrt(np.maximum(w, 0.0))
        inv = np.where(S > 0, 1.0 / np.where(S > 0, S, 1.0), 0.0)
        return U, S, (U.conj().T @ Xd) * inv[:, None]
    w, V = np.linalg.eigh(Xd.conj().T @ Xd)
    w, V = w[::-1], V[:, ::-1]
    S = np.sqrt(np.maximum(w, 0.0))
    inv = np.where(S > 0, 1.0 / np.where(S > 0, S, 1.0), 0.0)
    return (Xd @ V) * inv[None, :], S, V.conj().T

def opt_shrink_frob(vals2, M, N, sigma2):
    """Equivalent to opt_shrink_frob.m"""
 # vals2 = vals2 - 2*(N+M)*sigma2 + (N-M)^2*sigma2^2./vals2;
    return vals2 - 2 * (N + M) * sigma2 + (N - M)**2 * sigma2**2 / vals2

def opt_shrink_operator(vals2, M, N, sigma2):
    """Optimal shrinkage for operator-norm loss (Gavish & Donoho 2017), same units as
    opt_shrink_frob (squared singular values in, squared singular values out): the estimate of
    the clean squared singular value, ((l - s2(M+N)) + sqrt((l - s2(M+N))^2 - 4 s2^2 M N)) / 2.
    For a strong component it subtracts ~sigma2*(M+N), half of what the Frobenius rule
    subtracts; at the noise edge it keeps ~sigma2*sqrt(M*N) where the Frobenius rule gives 0."""
    t = vals2 - sigma2 * (M + N)
    disc = np.maximum(t**2 - 4.0 * sigma2**2 * M * N, 0.0)
    return np.where(t > 0, 0.5 * (t + np.sqrt(disc)), 0.0)

def subtract_mean(X):
    """Subtracts the mean along the longest dimension."""
    M, N = X.shape
    if M < N:
        X_mean = np.mean(X, axis=1, keepdims=True)
    else:
        X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    return X_centered, X_mean

def estimate_noise(S_diag, dims):
    """Marchenko-Pastur cutoff estimation."""
    M = len(S_diag)
    N = np.prod(dims) / M
    vals2 = S_diag**2
    P_vec = np.arange(len(vals2))
    
    # MATLAB: cumsum(vals2,'reverse') ./ (M-P) ./ (N-P)
    sigma2_estimates = np.cumsum(vals2[::-1])[::-1] / ((M - P_vec) * (N - P_vec))
    cutoff_estimates = sigma2_estimates * (np.sqrt(M) + np.sqrt(N))**2
    
    # find first element below cutoff
    idx = np.where(vals2 < cutoff_estimates)[0]
    if len(idx) == 0:
        P = len(vals2)
        sigma2 = 0.0
    else:
        P = idx[0]
        sigma2 = sigma2_estimates[P]
        
    if P == 0 and min(M, N) == 1:
        P = 1
        
    return sigma2, P

def combined_noise_estimate(S_list, dims, P_list):
    """Combined MP-PCA variance estimate across all unfoldings."""
    sigma2 = 0.0
    denominator = 0.0
    
    for n in range(len(S_list)):
        M = len(S_list[n])
        N = np.prod(dims) / M
        vals2 = S_list[n]**2
        # MATLAB: sum(vals2(P(n)+1:end))
        sigma2 += np.sum(vals2[P_list[n]:])
        denominator += (M - P_list[n]) * (N - P_list[n])
        
    sigma2 = sigma2 / denominator if denominator > 0 else 0.0
    
    P_out = np.zeros(len(S_list), dtype=int)
    for n in range(len(S_list)):
        M = len(S_list[n])
        N = np.prod(dims) / M
        cutoff = sigma2 * (np.sqrt(M) + np.sqrt(N))**2
        P_out[n] = np.count_nonzero((S_list[n]**2) > cutoff)
        
    return sigma2, P_out

def discard_noise_components(U, S_diag, V, sigma2):
    """Truncates matrices to P signal components."""
    M = U.shape[0]
    N = V.shape[0] 
    cutoff = sigma2 * (np.sqrt(M) + np.sqrt(N))**2
    P = np.count_nonzero(S_diag**2 > cutoff)
    
    U_new = U[:, :P]
    S_diag_new = S_diag[:P]
    V_new = V[:, :P]
    return U_new, S_diag_new, V_new, P

def apply_optimal_shrinkage(U, S_diag, V, sigma2, rule="frob"):
    """Applies optimal shrinkage to singular values ('frob' as in the original, or 'operator')."""
    if len(S_diag) == 0:
        return S_diag
    M = U.shape[0]
    N = V.shape[0]
    P = len(S_diag)
    vals2 = S_diag**2

    if rule == "operator":
        vals2 = opt_shrink_operator(vals2, max(M - P, 1), max(N - P, 1), sigma2)
    elif rule == "frob":
        vals2 = opt_shrink_frob(vals2, max(M - P, 1), max(N - P, 1), sigma2)
    else:
        raise ValueError(f"shrink_rule must be 'frob' or 'operator', got {rule!r}")
    # Ensure no negative values before sqrt (equivalent to real(sqrt(vals2)) in MATLAB for edge cases)
    return np.sqrt(np.maximum(vals2, 0.0))

def denoise_array_recursive_tensor(X, **kwargs):
    """Equivalent to denoise_array_recursive_tensor.m"""
    dims = X.shape
    
    options = {
        'opt_shrink': True,
        'subtract_mean': False,
        'num_inds': len(dims),
        'full_sigma2_pass': True,
        'test': False,
        'svd': 'lapack',  # 'gram' is faster; the v6 pipeline passes it explicitly
        'shrink_rule': 'frob',
        'num_stages': None
    }
    options.update(kwargs)
    
    # SVDs truncated: the first n_trunc index groups (all by default); shrinkage goes on the last
    num_inds = options['num_inds']
    if options['num_stages'] is None:
        n_trunc = num_inds
    else:
        n_trunc = max(1, min(int(options['num_stages']), num_inds))
    last_trunc = n_trunc - 1
    
    use_MPPCA = 'sigma2' not in kwargs
    if not use_MPPCA:
        sigma2 = kwargs['sigma2']
        options['use_initial_sigma2_pass'] = False
        
    # Handle special case of matrix input
    if len(dims) == 2:
        if min(dims) == 1:
            return X, 0.0, [1]
            
        if options['subtract_mean']:
            X_curr, X_mean = subtract_mean(X)
        else:
            X_curr = X
            X_mean = 0.0
            
        U, S_diag, Vh = _svd(X_curr, options['svd'])
        V = Vh.T
        
        if use_MPPCA:
            sigma2, P_val = estimate_noise(S_diag, dims)
            
        U, S_diag, V, P_val = discard_noise_components(U, S_diag, V, sigma2)
        
        if options['opt_shrink']:
            S_diag = apply_optimal_shrinkage(U, S_diag, V, sigma2, options['shrink_rule'])
            
        X_rec = (U * S_diag) @ V.T + X_mean
        return X_rec, sigma2, [P_val]

    U_list, S_list, V_list, X_mean_list = {}, {}, {}, {}
    P_list = []
    
    X_curr = X.copy()
    
    if use_MPPCA:
        num_SVDs = options['num_inds'] if options['full_sigma2_pass'] else 1
        for n in range(num_SVDs):
            # Flatten with order='F' matches MATLAB's mode-unfolding and automatic index rotation
            current_dim = dims[n]
            X_curr = X_curr.reshape((current_dim, -1), order='F')
            
            if options['subtract_mean']:
                X_curr, X_m = subtract_mean(X_curr)
                X_mean_list[n] = X_m
            else:
                X_mean_list[n] = 0.0
                
            U_curr, S_diag_curr, Vh_curr = _svd(X_curr, options['svd'])
            V_curr = Vh_curr.T
            
            _, p_val = estimate_noise(S_diag_curr, dims)
            P_list.append(p_val)
            
            U_list[n] = U_curr
            S_list[n] = S_diag_curr
            V_list[n] = V_curr
            
            # MATLAB: X = V{n}*S{n}
            X_curr = V_curr * S_diag_curr
            
        sigma2, P_list = combined_noise_estimate([S_list[i] for i in range(num_SVDs)], dims, P_list)
        P_list = list(P_list) # convert back to list for mutable assignments
    else:
        n = 0
        current_dim = dims[n]
        X_curr = X_curr.reshape((current_dim, -1), order='F')
        
        if options['subtract_mean']:
            X_curr, X_m = subtract_mean(X_curr)
            X_mean_list[n] = X_m
        else:
            X_mean_list[n] = 0.0
            
        U_curr, S_diag_curr, Vh_curr = _svd(X_curr, options['svd'])
        U_list[n] = U_curr
        S_list[n] = S_diag_curr
        V_list[n] = Vh_curr.T
        
    # Recursive SVD pass
    n = 0
    U_curr, S_diag_curr, V_curr, p_val = discard_noise_components(U_list[n], S_list[n], V_list[n], sigma2)
    U_list[n], S_list[n], V_list[n] = U_curr, S_diag_curr, V_curr
    if use_MPPCA:
        P_list[n] = p_val
    else:
        P_list = [p_val]
    # Only when num_stages is set explicitly can the first SVD be the last truncated one
    # (the default path is untouched, so existing results reproduce bit for bit).
    if options['opt_shrink'] and options['num_stages'] is not None and last_trunc == 0:
        S_diag_curr = apply_optimal_shrinkage(U_curr, S_diag_curr, V_curr, sigma2, options['shrink_rule'])
        S_list[n] = S_diag_curr
        
    X_curr = V_curr * S_diag_curr
    
    for n in range(1, options['num_inds']):
        if P_list[n-1] == 0:
            P_list = P_list[:n]
            break
            
        current_dim = dims[n]
        X_curr = X_curr.reshape((current_dim, -1), order='F')
        
        if options['subtract_mean']:
            X_curr, X_m = subtract_mean(X_curr)
            X_mean_list[n] = X_m
        else:
            X_mean_list[n] = 0.0
            
        U_curr, S_diag_curr, Vh_curr = _svd(X_curr, options['svd'])
        V_curr = Vh_curr.T
        
        if n < n_trunc:
            U_curr, S_diag_curr, V_curr, p_val = discard_noise_components(U_curr, S_diag_curr, V_curr, sigma2)
        else:
            p_val = len(S_diag_curr)   # not truncated: keep every component (lossless rotation)
        
        U_list[n], S_list[n], V_list[n] = U_curr, S_diag_curr, V_curr
        if len(P_list) > n:
            P_list[n] = p_val
        else:
            P_list.append(p_val)
            
        if options['opt_shrink'] and n == last_trunc:
            S_list[n] = apply_optimal_shrinkage(U_list[n], S_list[n], V_list[n], sigma2, options['shrink_rule'])
            
        X_curr = V_list[n] * S_list[n]
        
    # Reconstruct denoised X backwards
    for n in range(len(P_list)-1, -1, -1):
        if P_list[n] == 0:
            X_curr = np.zeros((U_list[n].shape[0], X_curr.shape[0])) + X_mean_list[n]
        else:
            tmp = X_curr.reshape((-1, P_list[n]), order='F')
            X_curr = U_list[n] @ tmp.T + X_mean_list[n]
            
    X_out = X_curr.reshape(dims, order='F')
    
    # Pad P if it broke early
    P_out = list(P_list) + [0] * (options['num_inds'] - len(P_list))
    
    return X_out, sigma2, P_out


def denoise_recursive_tensor(data, window, **kwargs):
    """
    Main entry point. Equivalent to denoise_recursive_tensor.m
    
    data: numpy ndarray
    window: tuple or list (e.g. (5, 5))
    kwargs: Optional name-value parameters (e.g. mask, indices, opt_shrink)
    """
    dims = data.shape
    dims_vox = tuple(dims[:len(window)])
    if len(dims_vox) == 1:
        dims_vox = (dims_vox[0], 1)
        
    num_vox = np.prod(dims_vox)
    # Using order='F' ensures we stride over memory identically to MATLAB
    data_flat = data.reshape((num_vox, -1), order='F')
    
    vox_indices_def = list(range(len(window)))
    mod_indices_def = list(range(len(window), len(dims)))
    default_indices = [vox_indices_def] + [[m] for m in mod_indices_def]
    
    options = {
        'indices': default_indices,
        'mask': np.ones(dims_vox, dtype=bool),
        'center_assign': False,
        'stride': [1]*len(window),
        'test': False,
        'progress': True
    }
    options.update(kwargs)
    show_progress = bool(options['progress'])
    
    indices = options['indices']
    stride = np.array(options['stride'])
    mask = options['mask']
    
    if mask.shape != dims_vox:
        raise ValueError('Mask dimensions do not match data window dimensions')
        
    array_size = list(window) + list(dims[len(window):])
    flat_indices = [idx for sublist in indices for idx in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
    max_idx = max(flat_indices) if flat_indices else -1
    
    if max_idx + 1 > len(dims):
        array_size += [1] * ((max_idx + 1) - len(dims))
        
    window_prod = np.prod(window)
    window_subs = np.unravel_index(np.arange(window_prod), window, order='F')
    index_increments = np.ravel_multi_index(window_subs, dims_vox, order='F')
    
    permute_order = [idx for sublist in indices for idx in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
    remaining_dims = [i for i in range(len(array_size)) if i not in permute_order]
    permute_order += remaining_dims
    
    new_size = []
    for group in indices:
        group_list = group if isinstance(group, (list, tuple)) else [group]
        sz = int(np.prod([array_size[i] for i in group_list]))
        new_size.append(sz)
        
    rem_sz = int(np.prod(array_size) / np.prod(new_size)) if new_size else int(np.prod(array_size))
    new_size.append(rem_sz)
    new_size = [s for s in new_size if s != 1]
    new_size.append(1)
    
    denoised = np.zeros_like(data_flat)
    count = np.zeros((num_vox, 1), dtype=float)
    Sigma2 = np.zeros((num_vox, 1), dtype=float)
    
    # Pre-allocate P, dynamically grow if needed
    P_max_len = len(indices) + 1
    P = np.zeros((num_vox, P_max_len), dtype=float)
    
    mask_flat = mask.ravel(order='F')
    center_ind = None
    if options['center_assign']:
        center_subs = [int(np.ceil(w / 2.0)) - 1 for w in window]
        center_ind = np.ravel_multi_index(center_subs, window, order='F')
        
    # Set up subset of kwargs for the recursive array function (pass anything that isn't specific to this sliding window func)
    kwargs_denoise = {k: v for k, v in kwargs.items() if k not in ['indices', 'mask', 'center_assign', 'stride', 'progress']}

    if show_progress:
        print(f'Denoising progress:   0%', end='', flush=True)

    for i in range(num_vox):
        index_vector = np.array(np.unravel_index(i, dims_vox, order='F'))
        
        if np.any(index_vector + np.array(window) > np.array(dims_vox)):
            continue
            
        if np.any((index_vector) % stride != 0):
            continue
            
        vox_indices_curr = i + index_increments
        
        maskX = mask_flat[vox_indices_curr]
        if np.count_nonzero(maskX) == 0:
            continue
            
        if options['center_assign']:
            if not maskX[center_ind]:
                continue
                
        X_patch = data_flat[vox_indices_curr, :]
        X_patch = X_patch.reshape(array_size, order='F')
        X_patch = np.transpose(X_patch, permute_order)
        X_patch = X_patch.reshape(new_size, order='F')
        
        # Core Denoise Step
        X_denoised, sigma2_val, p_val = denoise_array_recursive_tensor(
            X_patch, 
            num_inds=min(len(X_patch.shape), len(indices)),
            **kwargs_denoise
        )
        
        # Un-reshape and un-permute
        array_size_permuted = [array_size[idx] for idx in permute_order]
        X_denoised = X_denoised.reshape(array_size_permuted, order='F')
        
        inverse_permute = [0] * len(permute_order)
        for pos, idx in enumerate(permute_order):
            inverse_permute[idx] = pos
        X_denoised = np.transpose(X_denoised, inverse_permute)
        
        if options['center_assign']:
            X_denoised = X_denoised.reshape((len(vox_indices_curr), -1), order='F')
            X_denoised = X_denoised[center_ind:center_ind+1, :]
            vox_indices_assign = [vox_indices_curr[center_ind]]
        else:
            X_denoised = X_denoised.reshape((len(vox_indices_curr), -1), order='F')
            vox_indices_assign = vox_indices_curr
            
        denoised[vox_indices_assign, :] += X_denoised
        count[vox_indices_assign, 0] += 1
        Sigma2[vox_indices_assign, 0] += sigma2_val
        
        p_len = len(p_val)
        if P.shape[1] < p_len:
            P = np.pad(P, ((0, 0), (0, p_len - P.shape[1])), mode='constant')
        # Fix: accumulate ranks on the voxels this patch ASSIGNS to, matching `count`.
        # (Was vox_indices_curr: with center_assign=True every voxel summed the ranks of
        #  all ~prod(window) patches covering it but was divided by count = 1, inflating
        #  P_map and snr_gain. Denoised output and Sigma2 were never affected.)
        P[vox_indices_assign, :p_len] += p_val

        # Progress reporting
        if show_progress and (i % max(1, round(num_vox/100)) == 0 or i == num_vox - 1):
            sys.stdout.write(f'\b\b\b\b{round(100*(i+1)/num_vox):3d}%')
            sys.stdout.flush()

    if show_progress:
        print() # Newline post-progress
    
    skipped_vox = (count == 0).flatten()
    denoised[skipped_vox, :] = data_flat[skipped_vox, :]
    Sigma2[skipped_vox, :] = np.nan
    P[skipped_vox, :] = np.nan
    
    count[skipped_vox, :] = 1
    denoised = denoised / count
    Sigma2 = Sigma2 / count
    P = P / count
    
    # SNR Gain Estimate
    prod_new_size = np.prod(new_size)
    new_size_arr = np.array(new_size[:-1]) if len(new_size) > 1 else np.array([new_size[0]])
    
    if P.shape[1] == 1:
        SNR_gain = np.sqrt(prod_new_size / (P[:,0]**2 + np.sum((new_size_arr - P[:,0:1]) * P[:,0:1], axis=1)))
    else:
        P_rel = P[:, :len(new_size_arr)]
        prod_P = np.prod(P_rel, axis=1)
        sum_term = np.sum((new_size_arr - P_rel) * P_rel, axis=1)
        # Avoid division by zero warnings cleanly
        with np.errstate(divide='ignore', invalid='ignore'):
            SNR_gain = np.sqrt(prod_new_size / (prod_P + sum_term))
            
    # Reshape results identically to MATLAB
    denoised = denoised.reshape(dims, order='F')
    Sigma2 = Sigma2.reshape(dims_vox, order='F')
    P = P.reshape(list(dims_vox) + [P.shape[1]], order='F')
    SNR_gain = SNR_gain.reshape(dims_vox, order='F')
    
    return denoised, Sigma2, P, SNR_gain