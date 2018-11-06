import tensorflow as tf
import numpy as np
from data import parse_cora
from models import NodeClassificationDCNN
from params import Param
import matplotlib.pyplot as plt

cora_path = './data/cora'

def run_node_classification():
    A, X, Y = parse_cora()
    
    parameters = Param(node_num=X.shape[0], feature_num=X.shape[1], hop_num=4, 
    classes_num=Y.shape[1], stop_window_size=5, num_epochs=40, batch_size=64,
    activation='tanh', learning_rate=0.001)
    
    dcnn = NodeClassificationDCNN(A, parameters)

    indices = np.arange(parameters.node_num).astype('int32')
    np.random.shuffle(indices)

    train_indices = indices[:parameters.node_num//3]
    valid_indices = indices[parameters.node_num//3:(2 * parameters.node_num)//3]
    test_indices = indices[(2*parameters.node_num)//3:]

    dcnn.fit(X, Y, train_indices, valid_indices, test_indices)

if __name__ == '__main__':
    run_node_classification()