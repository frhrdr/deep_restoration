import numpy as np


def pca_whiten_mats(cov, n_to_drop=1):
    e_vals, e_vecs = np.linalg.eigh(cov)
    assert all(a <= b for a, b in zip(e_vals[:-1], e_vals[1:]))  # make sure vals are sorted ascending (they should be)
    print('sum of eigenvalues', e_vals.sum())
    print('smallest eigenvalue', np.min(e_vals))
    full_eigv = sum(e_vals)
    keep_eigv = sum(e_vals[n_to_drop:])

    print('kept eigv fraction: ', keep_eigv / full_eigv)
    print('number of negative eigenvals', np.sum(e_vals < 0))
    e_vals = e_vals[n_to_drop:]  # dismiss first eigenvalue due to mean subtraction. (or more)
    print('smallest kept eigenvalue:', e_vals[0])
    e_vecs = e_vecs[:, n_to_drop:]
    sqrt_vals = np.sqrt(np.maximum(e_vals, 0))
    whiten = np.diag(1. / sqrt_vals) @ e_vecs.T
    un_whiten = e_vecs @ np.diag(sqrt_vals)
    return whiten, un_whiten.T


def zca_whiten_mats(cov):
    U, S, _ = np.linalg.svd(cov)
    s = np.sqrt(S.clip(1.e-7))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whiten = np.dot(np.dot(U, s_inv), U.T)
    un_whiten = np.dot(np.dot(U, s), U.T)
    return whiten, un_whiten


def not_whiten_mats(cov):
    eye = np.eye(cov.shape[0])
    return eye, eye

# principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


# def pca_whiten(data):
#     cov = np.dot(data.T, data) / (data.shape[0] - 1)
#     whiten, unwhiten = pca_whiten_mats(cov)
#     data = np.dot(data, whiten.T)
#     return data, unwhiten


# def zca_whiten(data):
#     cov = np.dot(data.T, data) / (data.shape[0] - 1)
#     whiten, unwhiten = zca_whiten_mats(cov)
#     data = np.dot(data, whiten.T)
#     return data, unwhiten


# def pca_whiten_as_ica(data):
#     n_samples, n_features = data.shape
#     data = data.T
#     data -= data.mean(axis=-1)[:, np.newaxis]
#     u, d, _ = np.linalg.svd(data, full_matrices=False)
#     K = (u / d).T[:n_features]
#     data = np.dot(K, data)
#     data *= np.sqrt(n_samples)
#     return data.T, K


# def pca_whiten_as_pca(data):
#     n_samples, n_features = data.shape
#     u, s, v = np.linalg.svd(data, full_matrices=False)
#     max_abs_cols = np.argmax(np.abs(u), axis=0)
#     signs = np.sign(u[max_abs_cols, range(u.shape[1])])
#     u *= signs
#     u *= np.sqrt(n_samples)
#     v *= signs[:, np.newaxis]
#     rerotate = v / s[:, np.newaxis] * np.sqrt(n_samples)
#     return u, rerotate