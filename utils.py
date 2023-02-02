from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari, davies_bouldin_score
from numpy.linalg import inv as inverse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import PolynomialCountSketch, Nystroem
from sklearn.feature_extraction.text import TfidfTransformer
from time import time
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

warnings.filterwarnings('ignore')

from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io as io
import os

def datagen(dataset):
  if dataset in ['wiki', 'pubmed', 'computers', 'acm', 'dblp']: 
    data = io.loadmat(os.path.join('data', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    adj = data.get('W')
    if adj is not None:
      adj = adj.astype(float)
      if not sp.issparse(adj):
          adj = sp.csc_matrix(adj)
    
    if not sparse and sp.issparse(features):
        features = features.toarray()
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return adj, features, labels, n_classes

  if dataset == 'arxiv': 
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='data')
    graph = dataset[0] 
    data = graph[0]
    labels = graph[1].reshape(-1)

    
    features = data['node_feat']
    row_ind = data['edge_index'][0]
    col_ind = data['edge_index'][1]
    data = np.ones(len(row_ind))
    
    N = M = len(features)
    adj = csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
    adj = (adj + adj.T)
    
    n_classes = len(np.unique(labels))

    return adj, features, labels, n_classes
     

def preprocess_dataset(adj, features, row_norm=True, sym_norm=True, feat_norm='l2', tf_idf=False, sparse=False, alpha=1, beta=1):
    if sym_norm:
        adj = aug_normalized_adjacency(adj, True, alpha=alpha)
    if row_norm:
        adj = row_normalize(adj, True, alpha=beta)

    if tf_idf:
        features = TfidfTransformer(norm=feat_norm).fit_transform(features)
    else:
        features = normalize(features, feat_norm)
    
    if not sparse:
        features = features.toarray()
    return adj, features

def aug_normalized_adjacency(adj, add_loops=True, alpha=1):
    if add_loops:
        adj = adj + alpha*sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True, alpha=1):
    if add_loops:
        mx = mx + alpha * sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def convert_sparse_matrix_to_sparse_tensor(X):
      coo = X.tocoo()
      indices = np.mat([coo.row, coo.col]).transpose()
      return tf.SparseTensor(indices, coo.data, coo.shape)


def clustering_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    def ordered_confusion_matrix(y_true, y_pred):
      conf_mat = confusion_matrix(y_true, y_pred)
      w = np.max(conf_mat) - conf_mat
      row_ind, col_ind = linear_sum_assignment(w)
      conf_mat = conf_mat[row_ind, :]
      conf_mat = conf_mat[:, col_ind]
      return conf_mat

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def square_feat_map(z, c=1):
  polf = PolynomialFeatures(include_bias=True)
  x = polf.fit_transform(z)
  coefs = np.ones(len(polf.powers_))
  coefs[0] = c
  coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
  coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c) 
  return x * coefs

@tf.function
def convolve(feature, adj_normalized, power):
  for _ in range(power):
    feature = tf.sparse.sparse_dense_matmul(adj_normalized, feature)
  return feature

def run_model(H, c, k):
  H = StandardScaler(with_std=False).fit_transform(H)
  svd = TruncatedSVD(k)
  svd.fit(H.T)
  U = svd.components_.T

  Z = square_feat_map(U, c=c)
  r = Z.sum(0)
  D = Z @ r 
  Z_hat = Z / D[:,None]**.5
  
  svd = TruncatedSVD(k+1)
  svd.fit(Z_hat.T)
  Q = svd.components_.T[:,1:]
  P = KMeans(k).fit_predict(Q)  
  return P, Q

