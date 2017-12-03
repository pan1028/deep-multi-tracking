import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import BasicLSTMCell
import tensorflow.contrib.slim as slim

class AppearanceNetwork():
    # Define model parameters
    def __init__(self, args):
        self.args = args

        self.is_training = args.is_training
        self.weight_decay = 1e-8
        self.feature_dim = 128

        # Define rnn structure params
        self.lstm_num_cells = args.lstm_num_cells
        self.lstm_num_units = args.lstm_num_units

        # Define cnn structure params
        self.l2_normalize = args.l2_normalize

    def _batch_norm_fn(self, x, scope=None, reuse=None):
        if scope is None:
            scope = tf.get_variable_scope().name + "/bn"
        return slim.batch_norm(x, scope=scope, reuse=reuse)

    def create_link(
            self, incoming, network_builder, scope, nonlinearity=tf.nn.elu,
            weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
            regularizer=None, is_first=False, summarize_activations=True, reuse=None):
        if is_first:
            network = incoming
        else:
            network = self._batch_norm_fn(incoming, scope=scope + "/bn", reuse=reuse)
            network = nonlinearity(network)
            if summarize_activations:
                tf.summary.histogram(scope+"/activations", network)

        pre_block_network = network
        post_block_network = network_builder(pre_block_network, scope)

        incoming_dim = pre_block_network.get_shape().as_list()[-1]
        outgoing_dim = post_block_network.get_shape().as_list()[-1]
        if incoming_dim != outgoing_dim:
            assert outgoing_dim == 2 * incoming_dim, \
                "%d != %d" % (outgoing_dim, 2 * incoming)
            projection = slim.conv2d(
                incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
                scope=scope+"/projection", weights_initializer=weights_initializer,
                biases_initializer=None, weights_regularizer=regularizer, reuse=reuse)
            network = projection + post_block_network
        else:
            network = incoming + post_block_network
        return network


    def create_inner_block(
            self, incoming, scope, nonlinearity=tf.nn.elu,
            weights_initializer=tf.truncated_normal_initializer(1e-3),
            bias_initializer=tf.zeros_initializer(), regularizer=None,
            increase_dim=False, summarize_activations=True, reuse=None):
        n = incoming.get_shape().as_list()[-1]
        stride = 1
        if increase_dim:
            n *= 2
            stride = 2

        incoming = slim.conv2d(
            incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
            normalizer_fn=self._batch_norm_fn, weights_initializer=weights_initializer,
            biases_initializer=bias_initializer, weights_regularizer=regularizer,
            scope=scope + "/1", reuse=reuse)
        if summarize_activations:
            tf.summary.histogram(incoming.name + "/activations", incoming)

        incoming = slim.dropout(incoming, keep_prob=0.6)

        incoming = slim.conv2d(
            incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
            normalizer_fn=None, weights_initializer=weights_initializer,
            biases_initializer=bias_initializer, weights_regularizer=regularizer,
            scope=scope + "/2", reuse=reuse)
        return incoming


    def cnn_residual_block(self, incoming, scope, nonlinearity=tf.nn.elu,
                       weights_initializer=tf.truncated_normal_initializer(1e3),
                       bias_initializer=tf.zeros_initializer(), regularizer=None,
                       increase_dim=False, is_first=False,
                       summarize_activations=True, reuse=None):

        def network_builder(x, s):
            return self.create_inner_block(
                x, s, nonlinearity, weights_initializer, bias_initializer,
                regularizer, increase_dim, summarize_activations, reuse=reuse)

        return self.create_link(
            incoming, network_builder, scope, nonlinearity, weights_initializer,
            regularizer, is_first, summarize_activations, reuse=reuse)


    def cnn_network(self, incoming, num_classes=1501, reuse=None, l2_normalize=True,
                       create_summaries=True, weight_decay=1e-8):
        nonlinearity = tf.nn.elu
        conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
        conv_bias_init = tf.zeros_initializer()
        conv_regularizer = slim.l2_regularizer(self.weight_decay)
        fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
        fc_bias_init = tf.zeros_initializer()
        fc_regularizer = slim.l2_regularizer(self.weight_decay)

        def batch_norm_fn(x):
            return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

        network = incoming
        network = slim.conv2d(
            network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
            padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
            weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
            weights_regularizer=conv_regularizer, reuse=reuse)
        if create_summaries:
            tf.summary.histogram(network.name + "/activations", network)
            tf.summary.image("conv1_1/weights", tf.transpose(
                slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                             max_outputs=128)
        network = slim.conv2d(
            network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
            padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
            weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
            weights_regularizer=conv_regularizer, reuse=reuse)
        if create_summaries:
            tf.summary.histogram(network.name + "/activations", network)

        network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

        network = self.cnn_residual_block(
            network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False, is_first=True,
            summarize_activations=create_summaries, reuse=reuse)
        network = self.cnn_residual_block(
            network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=create_summaries, reuse=reuse)

        network = self.cnn_residual_block(
            network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=True,
            summarize_activations=create_summaries, reuse=reuse)
        network = self.cnn_residual_block(
            network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=create_summaries, reuse=reuse)

        network = self.cnn_residual_block(
            network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=True,
            summarize_activations=create_summaries, reuse=reuse)
        network = self.cnn_residual_block(
            network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
            conv_regularizer, increase_dim=False,
            summarize_activations=create_summaries, reuse=reuse)

        feature_dim = network.get_shape().as_list()[-1]
        # print("feature dimensionality: ", feature_dim)
        network = slim.flatten(network)

        network = slim.dropout(network, keep_prob=0.6)
        network = slim.fully_connected(
            network, feature_dim, activation_fn=nonlinearity,
            normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
            scope="fc1", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init, reuse=reuse)

        features = network

        if l2_normalize:
            # Features in rows, normalize axis 1.
            features = slim.batch_norm(features, scope="ball", reuse=reuse)
            feature_norm = tf.sqrt(
                tf.constant(1e-8, tf.float32) +
                tf.reduce_sum(tf.square(features), [1], keep_dims=True))
            features = features / feature_norm

            with slim.variable_scope.variable_scope("ball", reuse=reuse):
                weights = slim.model_variable(
                    "mean_vectors", (feature_dim, num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=1e-3),
                    regularizer=None)
                scale = slim.model_variable(
                    "scale", (num_classes, ), tf.float32,
                    tf.constant_initializer(0., tf.float32), regularizer=None)
                if create_summaries:
                    tf.summary.histogram("scale", scale)
                # scale = slim.model_variable(
                #     "scale", (), tf.float32,
                #     initializer=tf.constant_initializer(0., tf.float32),
                #     regularizer=slim.l2_regularizer(1e-2))
                # if create_summaries:
                #     tf.scalar_summary("scale", scale)
                scale = tf.nn.softplus(scale)

            # Each mean vector in columns, normalize axis 0.
            weight_norm = tf.sqrt(
                tf.constant(1e-8, tf.float32) +
                tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
            logits = scale * tf.matmul(features, weights / weight_norm)

        else:
            logits = slim.fully_connected(
                features, num_classes, activation_fn=None,
                normalizer_fn=None, weights_regularizer=fc_regularizer,
                scope="softmax", weights_initializer=fc_weight_init,
                biases_initializer=fc_bias_init)

        return features, logits


    def inference(self, input_image_patches, is_valid=None):
        # First part: cnn network to get all appearance features
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
            for i in range(self.lstm_num_cells + 1):
                image_patch = input_image_patches[:, i, :, :, :]
                image_patch = tf.squeeze(image_patch)
                                
                reuse = True
                if i == 0:
                    reuse = is_valid
                features, _ = self.cnn_network(
                    image_patch, l2_normalize=self.l2_normalize, reuse=reuse,
                    create_summaries=self.is_training, weight_decay=self.weight_decay)
                
                if i == 0:
                    # test_target_features = target_features
                    target_features = tf.expand_dims(features, 1)
                elif i == self.lstm_num_cells:
                    detection_features = features
                else:
                    target_features = tf.concat([target_features, tf.expand_dims(features, 1)], axis=1)

        # Second part: rnn network to take in all the apperance features
        with slim.variable_scope.variable_scope("lstm", reuse=is_valid) as scope:
            lstm_cell = BasicLSTMCell(self.lstm_num_units, state_is_tuple=False)
            lstm_output, final_state = rnn.dynamic_rnn(lstm_cell, target_features, dtype=tf.float32)

        # Final part: fully connecter layer with softmax activation
        fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
        fc_bias_init = tf.zeros_initializer()
        fc_regularizer = slim.l2_regularizer(self.weight_decay)
        with slim.variable_scope.variable_scope("fc_layer", reuse=is_valid) as scope:
            lstm_output = tf.squeeze(lstm_output[:, -1, :])
            
            concat_features = tf.concat([lstm_output, detection_features], axis=1)
            concat_features = slim.dropout(concat_features, keep_prob=0.6)
            
            fc_output = slim.fully_connected(
                concat_features, self.feature_dim, activation_fn=tf.nn.elu,
                normalizer_fn=self._batch_norm_fn, weights_regularizer=fc_regularizer,
                weights_initializer=fc_weight_init, biases_initializer=fc_bias_init
            )

            logits = slim.fully_connected(fc_output, 2, activation_fn=tf.nn.softmax,
                normalizer_fn=None, weights_regularizer=fc_regularizer,
                weights_initializer=fc_weight_init, biases_initializer=fc_bias_init
            )

            return fc_output, logits


    def cross_entropy_loss(self, logits, labels):
        with slim.variable_scope.variable_scope("loss"):
            logits = tf.cast(logits, dtype=tf.float64)
            labels = tf.cast(labels, dtype=tf.float64)

            epsilon = 10e-6
            labels = tf.stack([labels, 1- labels], axis=1)
            cross_entropy = tf.negative(labels * tf.log(logits + epsilon))
            return tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='xentropy_mean')