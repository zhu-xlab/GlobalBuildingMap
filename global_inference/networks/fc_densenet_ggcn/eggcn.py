import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


ps = 256
psa = ps * ps
ws = 3
A = np.zeros((psa, psa))
for i in range(0, psa):
    coly = i % ps
    rowy = i // ps
    if coly >= ws:
        colfa = -ws
    else:
        colfa = -coly
    if rowy >= ws:
        rowfa = -ws
    else:
        rowfa = -rowy

    colf = ps - coly
    rowf = ps - rowy
    if colf >= ws:
        colf = ws
    if rowf >= ws:
        rowf = ws
    for k in range(rowfa, rowf):
        for j in range(colfa, colf):
            A[i, i + j + ps * k] = 1
            A[i + j + ps * k, i] = 1
# np.save('adjm3.npy',A)
# print(np.count_nonzero(A))
# rowa,cola=np.where(A==1)
# adjc=np.ones(np.count_nonzero(A))

# adj = sp.coo_matrix((adjc, (rowa,cola)),shape=(psa,psa),dtype=np.float32)
# adj = normalize(adj+ sp.eye(adj.shape[0]))
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# sp.save_npz('adj72.npz',adj)
# print (A)
# adj_lists = defaultdict(set)

# a=np.argwhere(A == 1)
# print(A)
# print(a)
# np.save('my_file256.npy', a)
# read_dictionary=np.load('my_filea.npy')
# print(read_dictionary)
rowa, cola = np.where(A == 1)
adjc = np.ones(np.count_nonzero(A))

adj = sp.coo_matrix((adjc, (rowa, cola)), shape=(psa, psa), dtype=np.float32)
adj = normalize(adj + sp.eye(adj.shape[0]))
# adj = sparse_mx_to_torch_sparse_tensor(adj)
sp.save_npz("adjgcn7.npz", adj)
