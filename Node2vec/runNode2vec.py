import argparse
import os.path

import networkx as nx
from gensim.models import Word2Vec
from Node2vec import node2vec
from Tools.tools import match_userid
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def runNode2vec(cfg):
    print("*" * 80)
    print("Node2vec:")
    args = parse_args(cfg)
    print(args)
    node2vec_main(args)
    userid, emb, label = match_userid(cfg)
    return userid, emb, label

def parse_args(cfg):
    '''
	Parses the node2vec arguments.
	'''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default="./dataset/" + cfg.dataset_name + "/profiles.txt",
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default="./dataset/" + cfg.dataset_name + "/emb/useremb.txt",
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=cfg.d_model,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=cfg.p,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=cfg.q,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(args):
    '''
	Reads the input network in networkx.
	'''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=str, data=(('weight', float),), create_using=nx.Graph())
    else:
        G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.Graph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, args):
    '''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)
    model.wv.save_word2vec_format(args.output)

    return


def node2vec_main(args):
    '''
	Pipeline for representational learning for all nodes in a graph.
	'''
    if not os.path.exists(os.path.dirname(args.output)):
        os.mkdir(os.path.dirname(args.output))
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)
