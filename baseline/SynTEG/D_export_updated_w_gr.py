import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import numpy as np
from tqdm import tqdm
import os
import argparse

prefix = './data/datasets/synteg/'

# class Config(object):
#     def __init__(self):
#         self.lstm_dim = 384
#         self.n_layer = 3
#         self.batch_size = 256
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

def _array_float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _array_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


def serialize_example(word, condition):
    feature = {
        'word': _array_int64_feature(word),
        'condition': _array_float_feature(condition)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


class LSTM(tf.keras.Model):
    def __init__(self, lstm_dim, n_layer, batch_size): #
        super().__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x):
        for i in range(config.n_layer):
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)

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
                self.recurrent_kernel[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:],
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


class Embedding(tf.keras.Model):
    def __init__(self, lstm_dim, num_code,  max_num_visit, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(lstm_dim)
        self.linear2 = tf.keras.layers.Dense(lstm_dim)
        self.linear0 = tf.keras.layers.Dense(lstm_dim)
        self.encoding = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=num_code+2,
                                                                                    output_mode='binary')
        self.age = tf.keras.layers.Embedding(age_space, 32) #
        self.gender = tf.keras.layers.Embedding(gender_space, 8) #
        self.country = tf.keras.layers.Embedding(country_space, 16) #
        self.center = tf.keras.layers.Embedding(center_space, 32) #
        self.birth = tf.keras.layers.Embedding(birth_space, 64) #
        self.aidsmode = tf.keras.layers.Embedding(aidsmode_space, 16) #
        
        self.starter_age = tf.keras.layers.Embedding(age_space, 32) #
        self.starter_gender = tf.keras.layers.Embedding(gender_space, 8) #
        self.starter_country = tf.keras.layers.Embedding(country_space, 16) #
        self.starter_center = tf.keras.layers.Embedding(center_space, 32) #
        self.starter_birth = tf.keras.layers.Embedding(birth_space, 64) #
        self.starter_aidsmode = tf.keras.layers.Embedding(aidsmode_space, 16) #

        # self.interval = tf.keras.layers.Dense(128)
        self.interval = tf.keras.layers.Embedding(interval_space,128)
        self.num_code = num_code
        self.max_num_visit = max_num_visit

    def call(self, code, age, interval, gender, country, center, birth, aidsmode):
        n = code.shape[0]
        code = tf.reshape(code, [-1, code.shape[-1]])
        code = tf.reshape(self.encoding(code), [n, self.max_num_visit, self.num_code+2])[:, :, 1:]
        a = self.age(age)
        g = self.gender(gender) #
        r = self.country(country) #
        c = self.center(center) #
        e = self.birth(birth) #
        h = self.aidsmode(aidsmode) #
        # interval = self.interval(tf.expand_dims(tf.math.log(interval),-1))
        interval = self.interval(interval)

#         feature1 = tf.concat((tf.expand_dims(self.starter(age[:,0]),1), self.linear1(tf.concat((code, a, interval), axis=-1))),axis=1)
        feature1 = tf.concat((tf.expand_dims(self.linear0(tf.concat((self.starter_age(age[:,0]), self.starter_gender(gender[:,0]), self.starter_country(country[:,0]),\
                                                                      self.starter_center(center[:,0]), self.starter_birth(birth[:,0]), self.starter_aidsmode(aidsmode[:,0])\
                                                                        ), axis=-1)),axis=1), self.linear1(tf.concat((code, a, interval, g, r, c, e, h), axis=-1))),axis=1) #
        return feature1


class CodeModel(tf.keras.Model):
    def __init__(self, lstm_dim, n_layer, vec_dims, num_code, max_num_visit, batch_size, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space): #
        super().__init__()
        self.embeddings = Embedding(lstm_dim, num_code, max_num_visit, interval_space, age_space, gender_space, country_space, center_space, birth_space, aidsmode_space) #
        self.lstm = LSTM(lstm_dim, n_layer, batch_size) #
        self.mlp0 = tf.keras.layers.Dense(config.lstm_dim, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(config.lstm_dim)
        self.max_num_visit = max_num_visit
    ### This is the code for CodeModel, the conditional vector seems to be feature_vec
    @tf.function
    def call(self, code, age, interval, length, gender, country, center, birth, aidsmode): 
        # note that this function (replacing the one in stage1_updated_w_gr) does not have the normalization process, so that the returned tensors need to be normalized in the gan_updated2_w_gr after reading these out. 
        # fature contains the condition and code
        feature = self.embeddings(code, age, interval, gender, country, center, birth, aidsmode) #
        mask = tf.sequence_mask(length, self.max_num_visit)
        latent = tf.boolean_mask(self.lstm(feature)[:,:-1], mask)
        feature_vec = self.mlp1(self.mlp0(latent))

        sample = tf.boolean_mask(code, mask)
        return feature_vec, sample


def train():
    lengths = np.load(prefix + 'length.npy').astype(np.int32)
    features = np.load(prefix + 'code.npy').astype(np.int32) + 1
    intervals = np.load(prefix + 'interval.npy').astype(np.int32)
    ages = np.load(prefix + 'age.npy').astype(np.int32)
    genders = np.load(prefix + 'gender.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    countrys = np.load(prefix + 'country.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    centers = np.load(prefix + 'center.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    births = np.load(prefix + 'birth.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0
    aidsmode = np.load(prefix + 'mode.npy', allow_pickle = True).astype(np.int32)  # new id starts from 0

    order = np.arange(len(lengths))
    np.random.seed(0)  
    np.random.shuffle(order)

    length_train = lengths[order]
    feature_train = features[order]
    age_train = ages[order]
    gender_train = genders[order]
    country_train = countrys[order]
    center_train = centers[order]
    birth_train = births[order]
    aidsmode_train = aidsmode[order]
    interval_train = intervals[order]

    dataset_train = tf.data.Dataset.from_tensor_slices((feature_train,
                                                        age_train,
                                                        interval_train,
                                                        length_train,
                                                       gender_train,
                                                       country_train,
                                                       center_train,
                                                       birth_train,
                                                       aidsmode_train))
    parsed_dataset_train = dataset_train.batch(config.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)

    model = CodeModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit, config.batch_size, \
                      config.interval_space,\
                      config.age_space, config.gender_space, config.country_space, \
                        config.center_space, config.birth_space, config.aidsmode_space)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_prefix + "-" + args.b_ckpt).expect_partial()
    
    print("Start writing files.", flush=True)
    num_p = 0
    with tf.io.TFRecordWriter(f'./save/synteg/training_checkpoints_condition_updated_{args.b_ckpt}.tfrecord') as writer:
        for i, batch in tqdm(enumerate(parsed_dataset_train)):
            code, age, interval, length, gender, country, center, birth, aidsmode = batch
            length = tf.squeeze(length)
            ### The "condition" seems to be defined in here, from CodeModel
            condition_vector, vec = model(code, age, interval, length, gender, country, center, birth, aidsmode) #
            for x, y in zip(condition_vector.numpy(), vec.numpy()):
                example = serialize_example(y, x)
                writer.write(example)
            num_p += len(age)
            # if i % 50 == 49:
            #     print(int(i), flush=True)

    print("The total number of patients: %d" % num_p)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',"-g", type=str)
    parser.add_argument("--b_ckpt","-B",type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    NUM_GPU = 1
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    checkpoint_directory = "./save/synteg/training_checkpoints_stage1_code_updated_w_gr"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    config = Config()
    print(f"gender_space in config: {config.gender_space}")
    train()