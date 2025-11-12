import tensorflow as tf
import time
import os, argparse
import tensorflow_probability as tfp
import numpy as np
from B_stage1_code_updated_w_gr import CodeModel

from synteg_config import Config
        
class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_weight("bias",
                                      shape=[self.num_outputs])

    def call(self, x, y):
        return x * y + self.bias


class Generator(tf.keras.Model):
    def __init__(self, G_DIMS):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu) for dim in G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5,center=False, scale=False)] + \
                                 [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in G_DIMS[1:-1]]
        self.output_layer_code = tf.keras.layers.Dense(G_DIMS[-1], activation=tf.nn.sigmoid)
        self.condition_layer = tf.keras.layers.Dense(G_DIMS[0])
        self.pointwiselayer = PointWiseLayer(G_DIMS[0])

    def call(self, x, condition):
        h = self.dense_layers[0](x)
        x = self.pointwiselayer(self.batch_norm_layers[0](h), self.condition_layer(condition))
        for i in range(1, len(self.dense_layers)):
            h = self.dense_layers[i](x)
            h = self.batch_norm_layers[i](h)
            x += h
        x = self.output_layer_code(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self, D_DIMS):
        super(Discriminator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in D_DIMS]
        self.linear = tf.keras.layers.Dense(config.lstm_dim)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, condition):
        x = self.layer_norm_layers[0](self.dense_layers[0](x))
        for i in range(1,len(self.dense_layers)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        c = self.linear(x)
        x_vec = c / tf.math.sqrt(tf.reduce_sum(c ** 2, axis=-1, keepdims=True))
        x = self.output_layer(x) + tf.reduce_sum(condition * x_vec, axis=-1, keepdims=True)
        return x

class DiscriminatorTest(tf.keras.Model):
    def __init__(self, D_DIMS, num_code):
        super().__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in D_DIMS[:3]]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in D_DIMS[:3]]
        self.output_layer = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

    def call(self, x):
        # x = self.prep(x)[:, 1:]
        x = self.dense_layers[0](x)
        x = self.layer_norm_layers[0](x)
        for i in range(1, len(self.dense_layers)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        x = self.output_layer(x)
        return x


def train():
    checkpoint_directory = "./save/synteg/training_checkpoints_gan_updated2_" + args.serial
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    config = Config()
    config.num_code += 1
    print(config.num_code)
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_code], tf.int64),
        'condition': tf.io.FixedLenFeature([config.lstm_dim], tf.float32)
    }
    prep = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=config.num_code + 1,
                                                                       output_mode='binary')

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['word'], parsed['condition']

    dataset_train = tf.data.TFRecordDataset(f'./save/synteg/training_checkpoints_condition_updated_{args.b_ckpt}.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset_train = parsed_dataset_train.shuffle(4096 * 8, reshuffle_each_iteration=True).batch(config.gan_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    generator = Generator(config.G_DIMS)
    discriminator = Discriminator(config.D_DIMS)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator,
                                     discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)
    # checkpoint.restore(checkpoint_prefix + '-1').expect_partial()

    # Load the latest checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Initializing from scratch.")

    @tf.function
    def d_step(real, condition):
        mvn = tfd.MultivariateNormalDiag(scale_diag=0.05 * condition)
        condition_g = condition + tf.squeeze(mvn.sample())
        condition = condition / tf.math.sqrt(tf.reduce_sum(condition ** 2, axis=-1, keepdims=True))
        condition_g = condition_g / tf.math.sqrt(tf.reduce_sum(condition_g ** 2, axis=-1, keepdims=True))
        z = tf.random.normal(shape=[config.gan_batch_size, config.Z_DIM])

        epsilon = tf.random.uniform(
            shape=[config.gan_batch_size, 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, condition_g)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real, condition)
            fake_output = discriminator(synthetic, condition)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch([interpolate, condition])
                interpolate_output = discriminator(interpolate, condition)
            w_grad = t.gradient(interpolate_output, [interpolate, condition])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad[0]), 1) + tf.reduce_sum(tf.square(w_grad[1]), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            disc_loss = 10 * gradient_penalty + w_distance

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step(condition):
        mvn = tfd.MultivariateNormalDiag(scale_diag=0.05 * condition)
        condition_g = condition + tf.squeeze(mvn.sample())
        condition = condition / tf.math.sqrt(tf.reduce_sum(condition ** 2, axis=-1, keepdims=True))
        condition_g = condition_g / tf.math.sqrt(tf.reduce_sum(condition_g ** 2, axis=-1, keepdims=True))
        z = tf.random.normal(shape=[config.gan_batch_size, config.Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z, condition_g)

            fake_output = discriminator(synthetic, condition)

            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        word, condition = batch
        print("Condition shape:", condition.shape)

        real = tf.reshape(prep(tf.reshape(word, [config.gan_batch_size, config.max_num_code])),
                          [config.gan_batch_size, config.num_code + 1])[:, 1:]

        disc_loss, w_distance = d_step(real, condition)
        g_step(condition)
        return disc_loss, w_distance

    print('training start')
    print(f"serial: {args.serial}\n batch size:{config.gan_batch_size}\n B ckpt:f{args.b_ckpt}")

    for epoch in range(10000):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        step = 0.0
        for b in parsed_dataset_train:
            loss, w = train_step(b)
            total_loss += loss
            total_w += w
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
        print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch), flush = True)
        if epoch % 25 == 24:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("checkpoint saved at epoch %d" % (epoch), flush = True)


def gen(model, epoch):
    config = Config()
    checkpoint_directory = "./save/synteg/training_checkpoints_gan_updated2_" + str(model)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_code], tf.int64),
        'condition': tf.io.FixedLenFeature([config.lstm_dim], tf.float32)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['word'], parsed['condition']

    dataset_train = tf.data.TFRecordDataset(f'./save/synteg/training_checkpoints_condition_updated_{args.b_ckpt}.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
        4096 * 8)
    parsed_dataset_train = parsed_dataset_train.batch(config.gan_batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    generator = Generator(config.G_DIMS)

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_prefix + '-' + str(epoch)).expect_partial()

    @tf.function
    def train_step(batch):
        ## The condition is taken from batch
        _, condition = batch
        condition = condition / tf.math.sqrt(tf.reduce_sum(condition ** 2, axis=-1, keepdims=True))
        z = tf.random.normal(shape=[condition.shape[0], config.Z_DIM])
        synthetic = generator(z, condition)
        return tf.math.round(synthetic)

    x = np.arange(config.num_code, dtype='int32')
    res = []
    count = 0
    for b in parsed_dataset_train:
        batch_syn = train_step(b).numpy()
        for r in batch_syn:
            r = x[r == 1]
            if len(r) > config.max_num_code:
                count += 1
                continue
            res.append(np.pad(r, (0, config.max_num_code - len(r)), 'constant', constant_values=-1))
    print(count)
    np.save('test/syn' + str(model) + '_epoch' + str(epoch), res)


def test(model, epoch):
    checkpoint_d= "./save/synteg/training_checkpoints_stage1_code_updated"
    checkpoint_p = os.path.join(checkpoint_d, "ckpt")
    checkpoint_directory = "./save/synteg/training_checkpoints_filter_updated2_" + args.serial + '_epoch' + str(epoch)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    config = Config()
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_code], tf.int64),
        'condition': tf.io.FixedLenFeature([config.lstm_dim], tf.float32)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['word']
    
    dataset_real = tf.data.TFRecordDataset(f'./save/synteg/training_checkpoints_condition_updated_{args.b_ckpt}.tfrecord')
    dataset_real = dataset_real.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = np.load('test/syn' + str(model) + '_epoch' + str(epoch) + '.npy') + 1
    dataset_syn = tf.data.Dataset.from_tensor_slices(data)

    dataset = tf.data.Dataset.zip((dataset_real, dataset_syn))
    dataset_train = dataset.take(int(len(data)*0.8)).shuffle(4096*4, reshuffle_each_iteration=True).batch(
        config.gan_batch_size, drop_remainder=True)
    dataset_test = dataset.skip(int(len(data)*0.8))
    dataset_test = dataset_test.take(int(len(data) * 0.2)).batch(
        config.gan_batch_size)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    discriminator = DiscriminatorTest(config.D_DIMS, config.num_code)
    # prep = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=config.num_code + 1, output_mode='binary')
    #TODO: maybe I need to udpate code model to get conditinal tranining?
    ### Old codemodel:
    # m = CodeModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code-1, config.max_num_visit)

    ### New codemodel according to B
    m = CodeModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit, config.gan_batch_size, \
                  config.interval_space,\
                      config.age_space, config.gender_space, config.country_space, \
                        config.center_space, config.birth_space, config.mode_space)
    checkpoint = tf.train.Checkpoint(model=m)
    
    checkpoint.restore(checkpoint_p + '-' + args.b_ckpt).expect_partial()
    checkpoint2 = tf.train.Checkpoint(model=m, discriminator=discriminator)
    m1 = tf.keras.metrics.AUC(200)
    m2 = tf.keras.metrics.AUC(200)

    @tf.function
    def train_step(batch, is_training):
        real, syn = batch
        with tf.GradientTape() as disc_tape:
            pred_real = tf.squeeze(discriminator(m.fil(real)))
            pred_syn = tf.squeeze(discriminator(m.fil(syn)))
            # pred_real = tf.squeeze(discriminator(prep(real)[:,1:]))
            # pred_syn = tf.squeeze(discriminator(prep(syn)[:,1:]))
            disc_loss = tf.keras.metrics.binary_crossentropy(pred_real, tf.ones_like(
                pred_real)) + tf.keras.metrics.binary_crossentropy(pred_syn, tf.zeros_like(pred_syn))

        if is_training:
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables+m.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables+m.trainable_variables))
            m1.update_state(tf.ones_like(pred_real), pred_real)
            m1.update_state(tf.zeros_like(pred_syn), pred_syn)
        else:
            m2.update_state(tf.ones_like(pred_real), pred_real)
            m2.update_state(tf.zeros_like(pred_syn), pred_syn)
        return

    print('training start')
    for epoch in range(10000):
        m1.reset_states()
        m2.reset_states()
        start_time = time.time()
        for batch_sample in dataset_train:
            train_step(batch_sample, True)
        for batch_sample in dataset_test:
            train_step(batch_sample, False)
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, train = %f, test = %f, (%.2f)'
        if epoch % 10 == 9:
            checkpoint2.save(file_prefix=checkpoint_prefix)
        print(format_str % (epoch, m1.result().numpy(), m2.result().numpy(), duration_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',"-g", type=str, default="0")
    parser.add_argument('--serial', "-S", type=str)
    parser.add_argument("--b_ckpt", "-B",type=str)
    # parser.add_argument('--epoch', type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tfd = tfp.distributions
    config = Config()
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    train()
#     gen(args.serial, args.epoch)
#     test(args.serial, args.epoch)
