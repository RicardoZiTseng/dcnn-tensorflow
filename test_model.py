import tensorflow as tf
from models import NodeClassificationDCNN
from data import parse_cora
from params import Param

def test_model():
    A, X, Y = parse_cora()
    
    parameters = Param(node_num=X.shape[0], feature_num=X.shape[1], hop_num=2, 
    classes_num=Y.shape[1], stop_window_size=5, num_epochs=20, batch_size=64,
    activation='tanh', learning_rate=0.05)

    model = NodeClassificationDCNN(A,parameters)

    print('out_activate: ' + str(model.out_activate))
    print('out: ' + str(model.out))
    print('Y: ' + str(model.Y))
    print('cost: ' + str(model.cost))