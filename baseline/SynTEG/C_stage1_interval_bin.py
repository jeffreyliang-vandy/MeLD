import tensorflow as tf
import time
import os
import numpy as np
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
import argparse
from sklearn.model_selection import train_test_split

prefix = './data/datasets/synteg/'

# class Config(object):
#     def __init__(self):
#         self.lstm_dim = 384
#         self.n_layer = 3
#         self.batch_size = 128 * 2
#         self.vec_dims = [384, 384, 384]
#         self.max_num_visit = 265
#         self.num_code = 395 # the total number of distinct codes
#         self.age_space = 25 # check age.npy to derive the number of distinct integer ages from 0 and then add 1.
#         self.gender_space = 3  # check gender.npy
#         self.country_space = 6  # check country.npy
#         self.center_space = 9
#         self.birth_space = 14
#         self.aidsmode_space = 11
#         self.interval_space = 100
from synteg_config import Config



def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0], 1, inputs.shape[2]], dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1, inputs.shape[1], 1])
    return inputs * mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self, lstm_dim, n_layer, batch_size):
        super().__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(lstm_dim, batch_size, return_sequences=True) for _ in range(n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(n_layer)]

    def call(self, x, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for i in range(len(self.layer)):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, batch_size, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences, stateful=True)
        self.mask = None
        self.batch_size = batch_size #

    def set_mask(self, is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units, self.units * 4]), 0.2)
        else:
            self.mask = tf.ones([self.units, self.units * 4])

    def _process_batch(self, inputs, initial_state):
        if not self.time_major:
            inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = array_ops.expand_dims(input_h, axis=0)
        input_c = array_ops.expand_dims(input_c, axis=0)

        params = recurrent_v2._canonical_to_params(  # pylint: disable=protected-access
            weights=[
                self.kernel[:, :self.units],
                self.kernel[:, self.units:self.units * 2],
                self.kernel[:, self.units * 2:self.units * 3],
                self.kernel[:, self.units * 3:],
                self.recurrent_kernel[:, :self.units] * self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2] * self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3] * self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:] * self.mask[:, self.units * 3:],
            ],
            biases=[
                self.bias[:self.units],
                self.bias[self.units:self.units * 2],
                self.bias[self.units * 2:self.units * 3],
                self.bias[self.units * 3:self.units * 4],
                self.bias[self.units * 4:self.units * 5],
                self.bias[self.units * 5:self.units * 6],
                self.bias[self.units * 6:self.units * 7],
                self.bias[self.units * 7:],
            ],
            shape=self._vector_shape)

        outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            if self.time_major:
                output = outputs
            else:
                output = array_ops.transpose(outputs, perm=(1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]

    def reset_states(self, states=None):
        if nest.flatten(self.states)[0] is None:
            def create_state_variable(state):
                return K.zeros([self.batch_size] + tensor_shape.as_shape(state).as_list())

            self.states = nest.map_structure(
                create_state_variable, self.cell.state_size)
            if not nest.is_sequence(self.states):
                self.states = [self.states]
        elif states is None:
            for state, size in zip(nest.flatten(self.states),
                                   nest.flatten(self.cell.state_size)):
                K.set_value(state, np.zeros([self.batch_size] +
                                            tensor_shape.as_shape(size).as_list()))
        else:
            flat_states = nest.flatten(self.states)
            flat_input_states = nest.flatten(states)
            if len(flat_input_states) != len(flat_states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(flat_states)) + ' states, '
                                                         'but it received ' + str(len(flat_input_states)) +
                                 ' state values. Input received: ' + str(states))
            set_value_tuples = []
            for i, (value, state) in enumerate(zip(flat_input_states,
                                                   flat_states)):
                if value.shape != state.shape:
                    raise ValueError(
                        'State ' + str(i) + ' is incompatible with layer ' +
                        self.name + ': expected shape=' + str(
                            (self.batch_size, state)) + ', found shape=' + str(value.shape))
                set_value_tuples.append((state, value))
            K.batch_set_value(set_value_tuples)


class Embedding(tf.keras.Model):
    def __init__(self, lstm_dim, num_code, max_num_visit, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(lstm_dim)
        self.linear2 = tf.keras.layers.Dense(lstm_dim)
#         self.linear0 = tf.keras.layers.Dense(lstm_dim)
        self.encoding = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=num_code+2,
                                                                                    output_mode='binary')
        self.age = tf.keras.layers.Embedding(age_space, 32) #
        self.gender = tf.keras.layers.Embedding(gender_space, 8) #
        self.country = tf.keras.layers.Embedding(country_space, 16) #
        self.center = tf.keras.layers.Embedding(center_space, 32) #
        self.birth = tf.keras.layers.Embedding(birth_space, 64) #
        self.aidsmode = tf.keras.layers.Embedding(aidsmode_space, 16) #
        
        self.interval = tf.keras.layers.Dense(128)
        self.num_code = num_code
        self.max_num_visit = max_num_visit

    def call(self, code, age, interval, gender, country, center, birth, aidsmode):
        n = code.shape[0]
        code = tf.reshape(code, [-1, code.shape[-1]])
        code = tf.reshape(self.encoding(code), [n, self.max_num_visit, self.num_code + 2])[:, :, 1:]
        a = self.age(age)
        g = self.gender(gender) #
        r = self.country(country) #
        c = self.center(center)
        e = self.birth(birth)
        h = self.aidsmode(aidsmode)

        # interval = self.interval(tf.expand_dims(tf.math.log(interval),-1))
        interval = self.interval(tf.expand_dims(tf.math.abs(interval),-1))
        
        feature1 = self.linear1(tf.concat((code, a, interval, g, r, c, e, h), axis=-1))
        # tf.print(tf.shape(feature1)) # [128 265 384]
        feature2 = self.linear2(code)
        
        return feature1, feature2

    
    def gen(self, s_code, s_age, s_interval, s_gender, s_country, s_center, s_birth, s_aidsmode , next_code): # note these are step info, not the entire visit sequence, thus need expand dim.
        
        n = s_code.shape[0]
        code = tf.reshape(self.encoding(s_code), [n, 1, self.num_code+2])[:, :, 1:]  # batch_size, 1, self.num_code + 1
        next_code = self.encoding(next_code)[:, 1:] # batch_size, self.num_code + 1
        age = tf.expand_dims(self.age(s_age), 1)
        gender = tf.expand_dims(self.gender(s_gender), 1) #
        country = tf.expand_dims(self.country(s_country), 1) #
        center = tf.expand_dims(self.center(s_center), 1) #
        birth = tf.expand_dims(self.birth(s_birth), 1) #
        aidsmode = tf.expand_dims(self.aidsmode(s_aidsmode), 1) #

        # interval = tf.expand_dims(self.interval(tf.expand_dims(tf.math.log(s_interval),-1)),1)
        interval = tf.expand_dims(self.interval(tf.expand_dims(tf.math.abs(s_interval),-1)),1)
        feature1 = self.linear1(tf.concat((code, age, interval, gender, country, center, birth, aidsmode), axis=-1)) # batch_size, 1, lstm_dim
        # tf.print(tf.shape(feature1))
        feature2 = self.linear2(next_code) # batch_size, lstm_dim
        
        return feature1, feature2


def abs_glorot_uniform(shape, dtype=None, partition_info=None):
    return tf.math.abs(tf.keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))


class IntervalModel(tf.keras.Model):
    def __init__(self, lstm_dim, n_layer, vec_dims, num_code, max_num_visit, batch_size, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space):
        super().__init__()
        self.max_num_visit = max_num_visit
        self.interval_space = interval_space
        self.embeddings =  Embedding(lstm_dim, num_code, max_num_visit, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space) #
        self.lstm = LSTM(lstm_dim, n_layer, batch_size)
        self.linear = tf.keras.layers.Dense(lstm_dim)
        self.mlp0 = tf.keras.layers.Dense(lstm_dim, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(128)

        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in vec_dims]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in vec_dims]
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.interval1 = tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=abs_glorot_uniform,
                                              kernel_constraint=tf.keras.constraints.NonNeg())

        self.interval2 = [tf.keras.layers.Dense(i, activation=tf.nn.tanh, kernel_initializer=abs_glorot_uniform,
                                                 kernel_constraint=tf.keras.constraints.NonNeg()) for i in [128, 128]]
        self.interval3 = tf.keras.layers.Dense(1, activation=tf.nn.softplus, kernel_initializer=abs_glorot_uniform,
                                                 kernel_constraint=tf.keras.constraints.NonNeg())

    def call(self, code, age, interval, length, gender, country, center, birth, aidsmode, is_training=False):
        feature, code = self.embeddings(code, age, interval, gender, country, center, birth, aidsmode)  # # feature.shape: batch_size, 500, lstm_dim; code.shape: batch_size, 500, lstm_dim
        mask = tf.sequence_mask(length - 1, self.max_num_visit - 1)
        latent = tf.boolean_mask(self.lstm(feature, is_training)[:, :-1], mask)

        vec = self.linear(tf.boolean_mask(code[:, 1:], mask))
        for i in range(len(self.dense_layers)):
            if is_training:
                vec = tf.nn.dropout(vec, 0.15)
            h = self.dense_layers[i](vec)
            h = self.layer_norm_layers[i](h)
            vec += h
        feature_vec = tf.concat((latent, vec), axis=-1)
        if is_training:
            feature_vec = tf.nn.dropout(feature_vec, 0.15)
        feature_vec = self.ln(self.mlp1(self.mlp0(feature_vec)))

        # interval = tf.math.log(interval + tf.random.uniform(interval.shape, 0, 1))
        interval = tf.math.abs(interval + tf.random.uniform(interval.shape, 0, 1)) # 0-44, 0-45
        interval = tf.expand_dims(tf.boolean_mask(interval[:, 1:], mask), -1)
        with tf.GradientTape() as g:
            g.watch(interval)
            latent = feature_vec + self.interval1(interval)
            for i in range(2):
                latent = self.interval2[i](latent)
            output = self.interval3(latent)
        hazard = g.gradient(output, interval)
        loss = -tf.reduce_mean(tf.math.log(hazard + 1e-6) - output)
        return loss

    def gen(self, step_code, step_age, step_interval, gender, country, center, birth, aidsmode, next_code):
        step_feature, next_code = self.embeddings.gen(step_code, step_age, step_interval, gender, country, center, birth, aidsmode, next_code)
        latent = tf.squeeze(self.lstm(step_feature, is_training=False))

        vec = self.linear(next_code)
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](vec)
            h = self.layer_norm_layers[i](h)
            vec += h
        feature_vec = tf.concat((latent, vec), axis=-1)
        feature_vec = self.ln(self.mlp1(self.mlp0(feature_vec)))
        
        config = Config()
        
        test_interval = tf.expand_dims(tf.expand_dims(tf.math.abs(tf.range(0, config.interval_space + 1, dtype=tf.float32)), 1), 0)
        latent = tf.expand_dims(feature_vec, 1) + self.interval1(test_interval)
        for i in range(2):
            latent = self.interval2[i](latent)
        x = tf.squeeze(-tf.math.exp(-self.interval3(latent))) # 0-365 prob 365 prbs - 364 prbs
        probs = x[:, 1:] - x[:, :-1] # 0-44 (45 - 44) -> 44
        # print second 
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
        # dist = tfp.distributions.Categorical(probs=probs)
        # samples = dist.sample() + 1
        return probs


def train():
    config = Config()
    
    lengths = np.load(prefix + 'length.npy', allow_pickle = True).astype(np.int32)
    features = np.load(prefix + 'code.npy', allow_pickle = True).astype(np.int32) + 1
    ages = np.load(prefix + 'age.npy', allow_pickle = True).astype(np.int32)
    genders = np.load(prefix + 'gender.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    countrys = np.load(prefix + 'country.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    intervals = np.load(prefix + 'interval.npy', allow_pickle = True).astype(np.float32)
    # intervals = np.load(prefix + 'interval.npy', allow_pickle = True).astype(np.float32) + 1
    print("Max interval:", np.max(intervals)+1)
    centers = np.load(prefix + 'center.npy',allow_pickle = True).astype(np.int32)
    births = np.load(prefix + 'birth.npy',allow_pickle = True).astype(np.int32)
    aidsmode = np.load(prefix + 'mode.npy',allow_pickle = True).astype(np.int32)
    
    idx = np.arange(len(lengths))[lengths>1]
    train_idx, test_idx = train_test_split(idx,test_size=0.15, random_state=0) #
    # train_idx = np.load(prefix + 'train_idx.npy', allow_pickle = True)
    # test_idx = np.load(prefix + 'test_idx.npy', allow_pickle = True)
    length_test = lengths[test_idx]
    feature_test = features[test_idx]
    age_test = ages[test_idx]
    gender_test = genders[test_idx]
    country_test = countrys[test_idx]
    interval_test = intervals[test_idx]
    center_test = centers[test_idx]
    birth_test = births[test_idx]
    aidsmode_test = aidsmode[test_idx]


    length_train = lengths[train_idx]
    feature_train = features[train_idx]
    age_train = ages[train_idx]
    gender_train = genders[train_idx]
    country_train = countrys[train_idx]
    interval_train = intervals[train_idx]
    center_train = centers[train_idx]
    birth_train = births[train_idx]
    aidsmode_train = aidsmode[train_idx]

    dataset_train = tf.data.Dataset.from_tensor_slices((feature_train,
                                                        age_train,
                                                        interval_train,
                                                        length_train,
                                                        gender_train,
                                                        country_train,
                                                        center_train,
                                                        birth_train,
                                                        aidsmode_train)).shuffle(4096 * 12, reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.batch(config.batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    dataset_val = tf.data.Dataset.from_tensor_slices((feature_test,
                                                      age_test,
                                                      interval_test,
                                                      length_test,
                                                     gender_test,
                                                        country_test,
                                                        center_test,
                                                        birth_test,
                                                        aidsmode_test))
    parsed_dataset_val = dataset_val.batch(config.batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    del features, feature_train, feature_test

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model = IntervalModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit, config.batch_size,\
                           config.interval_space,\
                           config.age_space, config.gender_space, config.country_space,\
                           config.center_space, config.birth_space, config.aidsmode_space
                            )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Load the latest checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Initializing from scratch.")

    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    @tf.function
    def one_step(batch, is_training):
        code, age, interval, length, gender, country, center, birth, aidsmode = batch
        length = tf.squeeze(length)
        with tf.GradientTape() as tape:
            loss = model(code, age, interval, length, gender, country, center, birth, aidsmode, is_training)
        if is_training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print('training start', flush=True)
    test_loss_ = 10000.0
    for epoch in range(1000):
        start_time = time.time()
        train_loss = 0.0
        train_step = 0.0
        test_loss = 0.0
        test_step = 0.0
        for b in parsed_dataset_train:
            train_loss += one_step(b, True).numpy()
            train_step += 1
        for b in parsed_dataset_val:
            test_loss += one_step(b, False).numpy()
            test_step += 1
        duration_epoch = int(time.time() - start_time)
        format_str = 'epoch: %d, train_loss_d = %f, test_loss_d = %f (%d)'
        test_loss_tmp = test_loss / test_step
        print(format_str % (epoch, train_loss / train_step,
                            test_loss_tmp,
                            duration_epoch), flush=True)
        if epoch % 5 == 4 and test_loss_tmp < test_loss_:
            test_loss_ = test_loss_tmp
            checkpoint.save(file_prefix=checkpoint_prefix)
            print('CKPT saved at epoch %d' % epoch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',"-g", type=str,default=0)
    args = parser.parse_args()

    checkpoint_directory = "./save/synteg/training_checkpoints_stage1_interval_bin"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    NUM_GPU = 1

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    train()
