class Param(object):
    def __init__(self, node_num=2708, feature_num=1433, hop_num=3, 
    classes_num=7, stop_window_size=5, num_epochs=10, batch_size=64,
    activation='tanh', learning_rate=0.05):
        self.node_num = node_num
        self.feature_num = feature_num
        self.hop_num = hop_num
        self.classes_num = classes_num
        self.stop_window_size = stop_window_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.activation = activation
        self.learning_rate = learning_rate
        