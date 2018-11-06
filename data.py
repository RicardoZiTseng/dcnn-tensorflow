import numpy as np

def _parse_cora_features_labels():
    cora_path = './data/cora/'

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    with open(cora_path + 'cora.content', 'r') as f:
        idx = 0
        for line in f.readlines():
            items = line.strip().split('\t')

            id = items[0]
            # one-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(label)
            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = idx
            idx += 1
    
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    return features, labels, id2index

def parse_cora():
    cora_path = './data/cora/'

    features, labels, id2index = _parse_cora_features_labels()

    print('features\' shape, labels\' shape, number of nodes ->')
    print(features.shape, labels.shape, len(id2index))

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype=np.float32)

    with open(cora_path + 'cora.cites', 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            adj[id2index[items[0]]][id2index[items[1]]] = 1.0
            adj[id2index[items[1]]][id2index[items[0]]] = 1.0
    
    adj = np.asarray(adj, dtype=np.float32)

    return adj.astype('float32'), features.astype('float32'), labels.astype('float32')