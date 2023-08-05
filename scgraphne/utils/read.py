import os
import random

import h5py
import pandas as pd
import scipy as sp
import torch


def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sample(x, label, seed):
    x_sample = pd.DataFrame()
    for i in range(len(np.unique(label)) + 1):
        data = x[label == i,]
        data = pd.DataFrame(data)
        data = data.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
        data['label'] = i
        x_sample = x_sample.append(data, ignore_index=True)

    y = np.asarray(x_sample['label'], dtype='int')
    x_sample = np.asarray(x_sample.iloc[:, :-1])
    return x_sample, y


def sample10(x, xt, label, seed):
    x_sample = pd.DataFrame()
    xt_sample = pd.DataFrame()
    for i in range(len(np.unique(label))):
        data = x[label == i,]
        data = pd.DataFrame(data)
        data = data.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
        data['label'] = i
        x_sample = x_sample.append(data, ignore_index=True)

        datat = xt[label == i,]
        datat = pd.DataFrame(datat)
        datat = datat.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
        xt_sample = xt_sample.append(datat, ignore_index=True)

    y = np.asarray(x_sample['label'], dtype='int')
    x_sample = np.asarray(x_sample.iloc[:, :-1])
    return x_sample, xt_sample, y


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


import numpy as np
import scipy.sparse


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():  # pragma: no cover
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
