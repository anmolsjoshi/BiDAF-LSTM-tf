# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.rnn import RNNCell
import numpy as np


def matrix_weight_mul(matrix, weight):
    matrix_shape = matrix.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    mat_flatten = tf.reshape(matrix, [-1, matrix_shape[-1]])
    result = tf.matmul(mat_flatten, weight)
    return tf.reshape(result, [-1, matrix_shape[1], weight_shape[-1]])

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.rnn_cell_fw,
                self.rnn_cell_bw,
                inputs,
                input_lens,
                dtype=tf.float32
            )

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

        
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):
            # Get the attention scores (logits) e                           # (batch_size, num_keys, value_vec_size)
            values_t = tf.transpose(values, perm=[0, 2, 1])                 # (batch_size, value_vec_size, num_values)
            e = tf.matmul(keys, values_t)                                   # (batch_size, num_keys, num_values)

            # Calculate attention distribution
            attn_logits_mask = tf.expand_dims(values_mask, 1)               # (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(e, attn_logits_mask, 2)           # (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            print("output: ", output.get_shape().as_list())
            print("p: ", self.keep_prob.get_shape().as_list())
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDirectionalAttn(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size, num_values, num_keys):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        ## Note that in BiDAF, value_vec_size == key_vec_size
        tf.assert_equal(key_vec_size, value_vec_size)
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.num_values = num_values
        self.num_keys = num_keys

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          keys_to_values: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          values_to_keys: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDirectionalAttn"):
            ## Note that in BiDAF, value_vec_size == key_vec_size
            vec_size = self.value_vec_size

            W_c = tf.get_variable(
                "W_c",
                shape=[vec_size, 1],
                initializer=tf.contrib.layers.xavier_initializer(seed=1),
            )
            W_q = tf.get_variable(
                "W_q",
                shape=[vec_size, 1],
                initializer=tf.contrib.layers.xavier_initializer(seed=3),
            )
            W_cq = tf.get_variable(
                "W_cq",
                shape=[vec_size, 1],
                initializer=tf.contrib.layers.xavier_initializer(seed=5),
            )

            ##### Calculate similarity matrix S #####

            # Compute value matrix C
            c_ = tf.reshape(values, [-1, vec_size])                     # (batch_size * num_values, vec_size)
            C = tf.matmul(c_, W_c)                                      # (batch_size * num_values, 1)
            C = tf.reshape(C, [-1, self.num_values])                    # (batch_size, num_values)
            C = tf.expand_dims(C, 1)                                    # (batch_size, 1, num_values)

            # Compute key matrix Q
            q_ = tf.reshape(keys, [-1, vec_size])                       # (batch_size * num_keys, vec_size)
            Q = tf.matmul(q_, W_q)                                      # (batch_size * num_keys, 1)
            Q = tf.reshape(Q, [-1, self.num_keys])                      # (batch_size, num_keys)
            Q = tf.expand_dims(Q, 2)                                    # (batch_size, num_keys, 1)

            # Compute key-value matrix CQ
            c_t = tf.transpose(c_, perm=[1, 0])                         # (vec_size, batch_size * num_values)
            wc_ = tf.multiply(c_t, W_cq)                                # (vec_size, batch_size * num_values)
            wc_ = tf.reshape(wc_, [-1, vec_size, self.num_values])      # (batch_size, vec_size, num_values)
            CQ = tf.matmul(keys, wc_)                                   # (batch_size, num_keys, num_values)

            # Use broadcasting to build similarity matrix S
            S = C + Q + CQ                                              # (batch_size, num_keys, num_values)

            ##### Key-to-Value (C2Q) Attention ######

            # Apply softmax to get attention distribution over previous hidden states
            values_attn_logits_mask = tf.expand_dims(values_mask, 1)                        # (batch_size, 1, num_values)
            _, values_attn_dist = masked_softmax(S, values_attn_logits_mask, 2)             # (batch_size, num_keys, num_values)

            # Use attention distribution to take weighted sum of values
            keys_to_values = tf.matmul(values_attn_dist, values)                            # (batch_size, num_keys, vec_size)

            # Apply dropout
            keys_to_values = tf.nn.dropout(keys_to_values, self.keep_prob)                  # (batch_size, num_keys, vec_size)


            ##### Value-to-Key (Q2C) Attention ######

            # Take max of the corresponding row of the similarity matrix
            m = tf.reduce_max(S, axis=2, keep_dims=True)                                    # (batch_size, num_keys, 1)

            # Apply softmax to get attention distribution over previous hidden states
            keys_attn_logits_mask = tf.expand_dims(keys_mask, 1)                            # (batch_size, 1, num_keys)
            _, keys_attn_dist = masked_softmax(m, keys_attn_logits_mask, 2)                 # (batch_size, num_keys, 1)

            # Use attention distribution to take weighted sum of keys                       # (batch_size, num_keys, vec_size)
            values_to_keys = tf.matmul(keys_attn_dist, keys)                                # (batch_size, num_keys, vec_size)

            # Apply dropout
            values_to_keys = tf.nn.dropout(values_to_keys, self.keep_prob)                  # (batch_size, num_keys, vec_size)

            return values_attn_dist, keys_to_values, keys_attn_dist, values_to_keys


class BiRNN(object):
    """
    It feeds the input through a RNN and returns all the hidden states.

    This code uses a bidirectional LSTM, but you could experiment with other types of RNN.
    """
    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.
        Returns:
            out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiRNN"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.rnn_cell_fw,
                self.rnn_cell_bw,
                inputs,
                input_lens,
                dtype=tf.float32,
            )

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class BiRNN2(object):
    """
    It feeds the input through a RNN and returns all the hidden states.

    This code uses a bidirectional LSTM, but you could experiment with other types of RNN.
    """
    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.
        Returns:
            out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiRNN2"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.rnn_cell_fw,
                self.rnn_cell_bw,
                inputs,
                input_lens,
                dtype=tf.float32,
            )

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
