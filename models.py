import numpy as np
import tensorflow as tf
from util import A_to_diffusion_kernel
from util import multi_class_hinge_loss
from util import batch_matmul

class NodeClassificationDCNN(object):
    """ A DCNN model for node classification.
    
    A shallow model.

    (K, X) -> DCNN -> Dense -> Out
    """
    def __init__(self, A, parameters):
        self.params = parameters
        self.node_num = self.params.node_num
        self.hops = self.params.hop_num
        self._compute_diffusion_kernel(A)
        self._create_placeholder()
        self._create_layer_weight()
        self._register_model_layers()
        self._register_cost_and_optimizer()

    def _compute_diffusion_kernel(self, A):
        self.K = A_to_diffusion_kernel(A, self.hops)

    def _create_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[self.params.node_num, self.params.feature_num])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.params.classes_num])
        self.Pt = tf.placeholder(tf.float32, shape=[None, self.params.hop_num + 1, self.params.node_num])
    
    def _create_layer_weight(self):
        self.W = tf.get_variable(name='W', shape=[self.params.hop_num + 1, self.params.feature_num])

    def _register_model_layers(self):
        print(self.Pt, self.X)
        Apow_dot_X = batch_matmul(self.Pt, self.X)
        print(Apow_dot_X)
        Apow_dot_X_times_W = Apow_dot_X * self.W
        Z = tf.nn.tanh(Apow_dot_X_times_W)
        Z = tf.contrib.layers.flatten(Z)
        out = tf.contrib.layers.fully_connected(Z, self.params.classes_num)
        out_activate = tf.nn.softmax(out)
        self.out, self.out_activate = out, out_activate
    
    def _register_cost_and_optimizer(self):
        print("self.out: " + str(self.out), "self.Y: " + str(self.Y))
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.Y))
        self.cost = multi_class_hinge_loss(logits=self.out_activate, labels=self.Y)
        # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.cost)

    def fit(self, X, Y, train_indices, valid_indices, test_indices):
        print("Training model...")

        validation_losses = []
        validation_losses_window = np.zeros(self.params.stop_window_size)
        validation_losses_window[:] = float('+inf')
        init = tf.global_variables_initializer()

        correct_pred = tf.equal(tf.argmax(self.out_activate, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.params.num_epochs):
                train_loss = 0.0

                np.random.shuffle(train_indices)

                num_batch = self.node_num // self.params.batch_size

                for batch in range(num_batch):
                    start = batch * self.params.batch_size
                    end = min((batch+1) * self.params.batch_size, train_indices.shape[0])

                    if start < end:
                        loss, acc, _ = sess.run([self.cost, accuracy, self.optimizer], 
                        feed_dict={self.X: X, self.Y: Y[train_indices[start:end],:], 
                        self.Pt: self.K[train_indices[start:end],:,:]})
                        
                        train_loss += loss

                        print("Epoch {}/{}:".format(epoch+1, self.params.num_epochs) + \
                        "Minibatch loss = {:.4f}, ".format(loss) + "Training Accuracy = {:.2f}".format(acc))

                        # _ = sess.run(self.optimizer, feed_dict={self.X: X, self.Y: Y[train_indices[start:end],:], 
                        # self.Pt: self.K[train_indices[start:end],:,:]})
                
                train_loss /= num_batch

                valid_loss, valid_acc = sess.run([self.cost, accuracy], feed_dict={self.X: X, self.Y: Y[valid_indices,:], 
                        self.Pt: self.K[valid_indices,:,:]})
                train_acc = sess.run(accuracy, feed_dict={self.X: X, self.Y: Y[train_indices,:], 
                        self.Pt: self.K[train_indices,:,:]})

                print("=" * 40)

                print("Epoch {} mean training error is: {:.4f}".format(epoch+1, train_loss))
                print("Epoch {} training accuracy is: {:.2f}".format(epoch+1, train_acc))
                print("Epoch {} validation error is: {:.4f}".format(epoch+1, valid_loss))
                print("Epoch {} validation accuracy is: {:.2f}".format(epoch+1, valid_acc))

                validation_losses.append(valid_loss)

                if valid_loss >= validation_losses_window.mean():
                    print('Validation loss did not decrease. Stopping early.')
                    break

                validation_losses_window[epoch % self.params.stop_window_size] = valid_loss

            loss, acc = sess.run([self.cost, accuracy], feed_dict={self.X:X, 
                    self.Y: Y[test_indices,:], self.Pt: self.K[test_indices,:,:]})
            print("Testing error is: {:.4f}".format(loss))
            print("Testing accuracy is: {:.2f}".format(acc))