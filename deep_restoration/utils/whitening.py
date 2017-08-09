import numpy as np


def pca_whiten_mats(cov, n_to_drop=1):
    e_vals, e_vecs = np.linalg.eigh(cov)
    assert all(a <= b for a, b in zip(e_vals[:-1], e_vals[1:]))  # make sure vals are sorted ascending (they should be)
    print(e_vals.sum())
    print(e_vals)
    full_eigv = sum(e_vals)
    keep_eigv = sum(e_vals[n_to_drop:])

    print('kept eigv fraction: ', keep_eigv / full_eigv)
    print('number of negative eigenvals', np.sum(e_vals < 0))
    e_vals = e_vals[n_to_drop:]  # dismiss first eigenvalue due to mean subtraction.
    e_vecs = e_vecs[:, n_to_drop:]
    sqrt_vals = np.sqrt(np.maximum(e_vals, 0))
    whiten = np.diag(1. / sqrt_vals) @ e_vecs.T
    un_whiten = e_vecs @ np.diag(sqrt_vals)
    return whiten, un_whiten.T


def zca_whiten_mats(cov):
    U, S, _ = np.linalg.svd(cov)
    s = np.sqrt(S.clip(1.e-6))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whiten = np.dot(np.dot(U, s_inv), U.T)
    un_whiten = np.dot(np.dot(U, s), U.T)
    return whiten, un_whiten


def pca_whiten(data):
    cov = np.dot(data.T, data) / (data.shape[0] - 1)
    whiten, unwhiten = pca_whiten_mats(cov)
    data = np.dot(data, whiten.T)
    return data, unwhiten


def zca_whiten(data):
    cov = np.dot(data.T, data) / (data.shape[0] - 1)
    whiten, unwhiten = zca_whiten_mats(cov)
    data = np.dot(data, whiten.T)
    return data, unwhiten


def pca_whiten_as_ica(data):
    n_samples, n_features = data.shape
    data = data.T
    data -= data.mean(axis=-1)[:, np.newaxis]
    u, d, _ = np.linalg.svd(data, full_matrices=False)
    K = (u / d).T[:n_features]
    data = np.dot(K, data)
    data *= np.sqrt(n_samples)
    return data.T, K