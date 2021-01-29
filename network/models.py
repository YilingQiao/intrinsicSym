import tensorflow as tf
import numpy as np

from ops import *
import tf_util

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_transform_K(inputs, is_training, bn_decay=None, K=3):
    """ Transform Net, input is BxNx1xK gray image
        Return:
            Transformation matrix of size KxK """
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
   
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [-1, 1024])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [-1, K, K])
    return transform

def percep_net(point_cloud, is_training, num_eigen, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    num_point  = point_cloud.get_shape()[1].value
    
    input_image = tf.expand_dims(point_cloud, -1)

    K = point_cloud.get_shape()[2]
    out1 = tf_util.conv2d(input_image, 64, [1,K], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    out2 = tf_util.conv2d(out1, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    out3 = tf_util.conv2d(out2, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    out4 = tf_util.conv2d(out3, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    out5 = tf_util.conv2d(out4, 4096, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    out_max = tf_util.max_pool2d(out5, [num_point,1], padding='VALID', scope='maxpool')

    # classification network
    net = tf.reshape(out_max, [-1, 4096])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='cla/fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp0')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='cla/fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp1')
    #TODO: how to set the num of neurons
    #TODO: Do we need BN?
    net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='cla/fc3', bn_decay=bn_decay)
    my_sign = tf_util.fully_connected(net, 2, bn=False, is_training=is_training, activation_fn=None, scope='cla/fc5', bn_decay=bn_decay)
    #TODO: what if add a sigmoid function?
    return my_sign


# def fmnet_model(phase, input_feature, model_evecs, model_evecs_trans, model_constraint, model_reference_C):
def SignNet(phase, input_feature, refer_sign):
    # C = percep_net(input_feature, phase, FLAGS.num_evecs)
    sign = percep_net(input_feature, phase, FLAGS.num_evecs)
    # cross entropy loss
    loss_sign = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sign, labels=refer_sign))
    net_loss = loss_sign
    tf.summary.scalar('net_loss', net_loss)
    merged = tf.summary.merge_all()

    return net_loss, sign, merged