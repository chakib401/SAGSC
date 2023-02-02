import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch, Nystroem
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import kneighbors_graph
from time import time
from scipy import sparse
from sklearn.metrics import silhouette_score
from utils import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Parameters
flags.DEFINE_string('dataset', 'acm', 'Name of the graph dataset (`acm`, `dblp`, `arxiv`, `pubmed` or `wiki`).')
flags.DEFINE_integer('power', 2, 'Propagation order.')
flags.DEFINE_integer('runs', 5, 'Number of runs per power.')


dataset = flags.FLAGS.dataset
p = flags.FLAGS.power
n_runs = flags.FLAGS.runs


adj, features, labels, n_classes =  datagen(dataset)
norm_adj, features = preprocess_dataset(adj, features, 
                                      tf_idf=True,
                                      sparse=True)


features = features.toarray()
n, d = features.shape
k = n_classes


metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['time'] = []



norm_adj = convert_sparse_matrix_to_sparse_tensor(norm_adj.astype('float64'))
features = tf.convert_to_tensor(features.astype('float64'))
x = features

for run in range(n_runs):
    features = x
    t0 = time()
    features = convolve(features, norm_adj, p)
    P, Q = run_model(features, c=2**-.5, k=k)

    metrics['time'].append(time()-t0)
    metrics['acc'].append(clustering_accuracy(labels, P)*100)
    metrics['nmi'].append(nmi(labels, P)*100)
    metrics['ari'].append(ari(labels, P)*100)

results = {
      'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items() }, 
      'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
    }

means = results['mean']
std = results['std']


print(f"{dataset} {p}")
print(f"{means['acc']}±{std['acc']} & {means['nmi']}±{std['nmi']} & {means['ari']}±{std['ari']}", sep=',')
print(f"{means['time']}±{std['time']}")