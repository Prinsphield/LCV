"""Definining network in this file."""
import tensorflow as tf
import math

import sys
sys.path.append('../')
from data_augmentation import flow_resize
from utils import lrelu
from warp import tf_warp
slim = tf.contrib.slim


def feature_extractor(x,
                      trainable=True,
                      reuse=None,
                      regularizer=None,
                      name='feature_extractor'):
  """Feature extractor."""
  with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=lrelu,
                        kernel_size=3,
                        padding='SAME',
                        trainable=trainable):
      net = {}
      net['conv1_1'] = slim.conv2d(x, 16, stride=2, scope='conv1_1')
      net['conv1_2'] = slim.conv2d(
          net['conv1_1'], 16, stride=1, scope='conv1_2')

      net['conv2_1'] = slim.conv2d(
          net['conv1_2'], 32, stride=2, scope='conv2_1')
      net['conv2_2'] = slim.conv2d(
          net['conv2_1'], 32, stride=1, scope='conv2_2')

      net['conv3_1'] = slim.conv2d(
          net['conv2_2'], 64, stride=2, scope='conv3_1')
      net['conv3_2'] = slim.conv2d(
          net['conv3_1'], 64, stride=1, scope='conv3_2')

      net['conv4_1'] = slim.conv2d(
          net['conv3_2'], 96, stride=2, scope='conv4_1')
      net['conv4_2'] = slim.conv2d(
          net['conv4_1'], 96, stride=1, scope='conv4_2')

      net['conv5_1'] = slim.conv2d(
          net['conv4_2'], 128, stride=2, scope='conv5_1')
      net['conv5_2'] = slim.conv2d(
          net['conv5_1'], 128, stride=1, scope='conv5_2')

      net['conv6_1'] = slim.conv2d(
          net['conv5_2'], 192, stride=2, scope='conv6_1')
      net['conv6_2'] = slim.conv2d(
          net['conv6_1'], 192, stride=1, scope='conv6_2')

  return net


def context_network(x,
                    flow,
                    trainable=True,
                    reuse=None,
                    regularizer=None,
                    name='context_network'):
  """Context network."""
  x_input = tf.concat([x, flow], axis=-1)
  with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=lrelu,
                        kernel_size=3,
                        padding='SAME',
                        trainable=trainable):
      net = {}
      net['dilated_conv1'] = slim.conv2d(
          x_input, 128, rate=1, scope='dilated_conv1')
      net['dilated_conv2'] = slim.conv2d(
          net['dilated_conv1'], 128, rate=2, scope='dilated_conv2')
      net['dilated_conv3'] = slim.conv2d(
          net['dilated_conv2'], 128, rate=4, scope='dilated_conv3')
      net['dilated_conv4'] = slim.conv2d(
          net['dilated_conv3'], 96, rate=8, scope='dilated_conv4')
      net['dilated_conv5'] = slim.conv2d(
          net['dilated_conv4'], 64, rate=16, scope='dilated_conv5')
      net['dilated_conv6'] = slim.conv2d(
          net['dilated_conv5'], 32, rate=1, scope='dilated_conv6')
      net['dilated_conv7'] = slim.conv2d(
          net['dilated_conv6'],
          2,
          rate=1,
          activation_fn=None,
          scope='dilated_conv7')

  refined_flow = net['dilated_conv7'] + flow

  return refined_flow


def get_shape(x, train=True):
  if train:
    x_shape = x.get_shape().as_list()
  else:
    x_shape = tf.shape(x)
  return x_shape


def estimator(x1,
              x2,
              flow,
              train=True,
              trainable=True,
              reuse=None,
              regularizer=None,
              name='estimator'):
  """Estimator network."""
  # warp x2 according to flow
  x_shape = get_shape(x1, train=train)
  height = x_shape[1]
  width = x_shape[2]
  # channel = x_shape[3]
  channel = x1.get_shape().as_list()[-1]
  x2_warp = tf_warp(x2, flow, height, width)

  # ---------------cost volume-----------------
  # normalize
  x1 = tf.nn.l2_normalize(x1, axis=3)
  x2_warp = tf.nn.l2_normalize(x2_warp, axis=3)
  d = 9
  x2_patches = tf.extract_image_patches(
      x2_warp, [1, d, d, 1],
      strides=[1, 1, 1, 1],
      rates=[1, 1, 1, 1],
      padding='SAME')
  x2_patches = tf.reshape(x2_patches, [-1, height, width, d, d, channel])
  x1_reshape = tf.reshape(x1, [-1, height, width, 1, 1, channel])

  # get symmetric positive definite kernel matrix
  with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
    # obtain orthogonal matrix
    raw_P = tf.get_variable(
        'raw_P',
        shape=[channel, channel],
        initializer=tf.keras.initializers.Identity(),
        dtype=tf.float32)
    raw_P_upper = tf.matrix_band_part(raw_P, 0, -1)
    skew_P = (raw_P_upper - tf.transpose(raw_P_upper)) / 2
    # Cayley transformation, W is in Special Orthogonal Group SO(n)
    P = tf.matmul((tf.eye(channel) + skew_P), tf.linalg.inv(tf.eye(channel) - skew_P))

    # obtain the diagonal matrix with positive numbers
    raw_D = tf.get_variable(
        'raw_D',
        shape=[channel,],
        initializer=tf.zeros_initializer(),
        dtype=tf.float32)
    trans_D = tf.atan(raw_D) * 2 / math.pi
    D = tf.matrix_diag(tf.div(1 + trans_D, 1 - trans_D))

    # the symmetric positive definite kernal matrix is
    W = tf.matmul(tf.matmul(tf.transpose(P), D), P)

    x1_dot_x2 = tf.multiply(tf.tensordot(x1_reshape, W, axes=[-1,0]), x2_patches)
    cost_volume = tf.reduce_sum(x1_dot_x2, axis=-1)
    cost_volume = tf.reshape(cost_volume, [-1, height, width, d * d])

  # --------------estimator network-------------
  net_input = tf.concat([cost_volume, x1, flow], axis=-1)
  with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=lrelu,
                        kernel_size=3,
                        padding='SAME',
                        trainable=trainable):
      net = {}
      net['conv1'] = slim.conv2d(net_input, 128, scope='conv1')
      net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
      net['conv3'] = slim.conv2d(net['conv2'], 96, scope='conv3')
      net['conv4'] = slim.conv2d(net['conv3'], 64, scope='conv4')
      net['conv5'] = slim.conv2d(net['conv4'], 32, scope='conv5')
      net['conv6'] = slim.conv2d(
          net['conv5'], 2, activation_fn=None, scope='conv6')
  # flow_estimated = net['conv6']
  return net


def _pyramid_processing(x1_feature,
                        x2_feature,
                        img_size,
                        train=True,
                        trainable=True,
                        reuse=None,
                        regularizer=None,
                        is_scale=True):
  """Pyramid processing given two features."""
  x_shape = tf.shape(x1_feature['conv6_2'])
  initial_flow = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2],
                          dtype=tf.float32,
                          name='initial_flow')
  flow_estimated = {}
  flow_estimated['level_6'] = estimator(
      x1_feature['conv6_2'],
      x2_feature['conv6_2'],
      initial_flow,
      train=train,
      trainable=trainable,
      reuse=reuse,
      regularizer=regularizer,
      name='estimator_level_6')['conv6']

  for i in range(4):
    feature_name = 'conv%d_2' % (5 - i)
    feature_size = tf.shape(x1_feature[feature_name])[1:3]
    initial_flow = flow_resize(
        flow_estimated['level_%d' % (6 - i)], feature_size, is_scale=is_scale)
    if i == 3:
      estimator_net_level_2 = estimator(
          x1_feature[feature_name],
          x2_feature[feature_name],
          initial_flow,
          train=train,
          trainable=trainable,
          reuse=reuse,
          regularizer=regularizer,
          name='estimator_level_%d' % (5 - i))
      flow_estimated['level_2'] = estimator_net_level_2['conv6']
    else:
      flow_estimated['level_%d' % (5 - i)] = estimator(
          x1_feature[feature_name],
          x2_feature[feature_name],
          initial_flow,
          train=train,
          trainable=trainable,
          reuse=reuse,
          regularizer=regularizer,
          name='estimator_level_%d' % (5 - i))['conv6']

  x_feature = estimator_net_level_2['conv5']
  flow_estimated['refined'] = context_network(
      x_feature,
      flow_estimated['level_2'],
      trainable=trainable,
      reuse=reuse,
      regularizer=regularizer,
      name='context_network')
  flow_estimated['full_res'] = flow_resize(
      flow_estimated['refined'], img_size, is_scale=is_scale)

  return flow_estimated


def pyramid_processing(batch_img1,
                       batch_img2,
                       train=True,
                       trainable=True,
                       regularizer=None,
                       is_scale=True):
  """Pyramid processing given two images."""
  img_size = tf.shape(batch_img1)[1:3]
  x1_feature = feature_extractor(
      batch_img1,
      trainable=trainable,
      regularizer=regularizer,
      name='feature_extractor')
  x2_feature = feature_extractor(
      batch_img2,
      trainable=trainable,
      reuse=True,
      regularizer=regularizer,
      name='feature_extractor')
  flow_estimated = _pyramid_processing(
      x1_feature,
      x2_feature,
      img_size,
      train=train,
      trainable=trainable,
      regularizer=regularizer,
      is_scale=is_scale)
  return flow_estimated


def pyramid_processing_bidirection(batch_img1,
                                   batch_img2,
                                   train=True,
                                   trainable=True,
                                   reuse=None,
                                   regularizer=None,
                                   is_scale=True):
  """Bidirectional pyramid processing given two images."""
  img_size = tf.shape(batch_img1)[1:3]
  x1_feature = feature_extractor(
      batch_img1,
      trainable=trainable,
      reuse=reuse,
      regularizer=regularizer,
      name='feature_extractor')
  x2_feature = feature_extractor(
      batch_img2,
      trainable=trainable,
      reuse=True,
      regularizer=regularizer,
      name='feature_extractor')

  flow_fw = _pyramid_processing(
      x1_feature,
      x2_feature,
      img_size,
      train=train,
      trainable=trainable,
      reuse=reuse,
      regularizer=regularizer,
      is_scale=is_scale)
  flow_bw = _pyramid_processing(
      x2_feature,
      x1_feature,
      img_size,
      train=train,
      trainable=trainable,
      reuse=True,
      regularizer=regularizer,
      is_scale=is_scale)
  return flow_fw, flow_bw

