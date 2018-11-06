import numpy as np
from data import parse_cora
import tensorflow as tf

def A_to_diffusion_kernel(A, k, add_one=True):
    """
    Computes [A**0, A**1, A**2, ..., A**k]
    
    :param A: 2d numpy array
    :param k: degree of series
    :return: 3d numpy array [A**0, A**1, A**2, ..., A**k]
    """

    assert k >= 0

    Apow = [np.identity(A.shape[0])]

    if k > 0:
        d = A.sum(0)

        if add_one:
            Apow.append(A / (d + 1.0))
            for i in range(2, k+1):
                Apow.append(np.dot(A / (d + 1.0), Apow[-1]))
        else:
            Apow.append(A / d)
            for i in range(2, k+1):
                Apow.append(np.dot(A / d, Apow[-1]))
    
    return np.transpose(np.asarray(Apow, dtype=np.float32), (1, 0, 2))

def test_A_to_diffusion_kernel():
    adj, features, labels = parse_cora()
    print('adj\'s shape is %s' % str(adj.shape))
    P = A_to_diffusion_kernel(adj, 5)
    print('Hop is 5, Diffusion kernel of adj\'s shape should be (2708, 6, 2708), so it\'s real shape is %s.'%str(P.shape))


def multi_class_hinge_loss(logits, labels):
    """
    logits' shape is (?, classes)
    labels' shape is (?, classes)
    """
    logits = tf.transpose(logits)
    labels = tf.transpose(labels)
    H = tf.reduce_max(logits * (1 - labels), 0)    
    L = tf.nn.relu((1 - logits + H) * labels)
    loss = tf.reduce_mean(tf.reduce_max(L, 0))
    return loss

def test_multi_class_hinge_loss():
    labels = tf.constant([[1,0,0],[0,1,0],[0,0,1]],dtype=tf.float32)
    logits = tf.constant([[3.2,5.1,-1.7],[1.3,4.9,2.0],[2.2,2.5,-3.1]])
    loss = multi_class_hinge_loss(logits,labels)
    sess = tf.Session()
    print('True loss is 3.1666667')
    print('Testing result is',sess.run(loss))

def batch_matmul(A, B):
    """
    @params:
    tensor A's shape is (batch_size, m, n)
    tensor B's shape is (n, k)

    @return:
    tensor C's shape is (batch_size, m, k)
    """
    C = tf.einsum('ijk,kl->ijl',A,B)
    return C

def test_batch_matmul():
    A = tf.constant(np.random.randn(3, 4, 6))
    B = tf.constant(np.random.randn(6, 7))
    C = batch_matmul(A, B)

    if C.shape == (3, 4, 7):
        print("Clear.")
    print(C)

test_multi_class_hinge_loss()