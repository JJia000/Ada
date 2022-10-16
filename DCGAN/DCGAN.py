import math
import os
import tensorflow.compat.v1 as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

class get_default_params():
    def __init__(self, batch_size, cfg):
        self.z_dim = cfg.z_dim
        self.init_conv_size = 1
        self.g_channels = [64, 32, 1]
        self.d_channels = [64, 128, 256]
        self.batch_size = batch_size
        self.learning_rate = cfg.learning_rate
        self.beta1 = 0.5
        self.fake_user_size = int(math.sqrt(cfg.d_model))


class UserData(object):
    def __init__(self, user_train, z_dim, fake_user_size):
        self._data = user_train
        self._example_num = len(self._data)
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0
        self._resize_fakeuser(fake_user_size)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)
        self._z_data = np.array(self._z_data)[p]
        self._data = np.array(self._data)[p]

    def _resize_fakeuser(self, fake_user_size):
        data = np.array(self._data).reshape((self._example_num, 1, fake_user_size, fake_user_size))
        data = data.transpose((0, 2, 3, 1))
        self._data = data

    def next_batch(self, ):
        self._random_shuffle()
        return self._data, self._z_data


def conv2d_transpose(inputs, out_channel, name, training, with_bn_relu=True):
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs,
                                                  out_channel,
                                                  [3, 3],
                                                  strides=(2, 2),
                                                  padding='SAME')
        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans, training=training)
            relu = tf.nn.relu(bn)
            return relu
        else:
            return conv2d_trans


def conv2d(inputs, out_channel, name, training):
    def leaky_relu(x, leak=0.2, name=''):
        return tf.maximum(x, x * leak, name=name)

    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs,
                                         out_channel,
                                         [3, 3],
                                         strides=(2, 2),
                                         padding='SAME')
        bn = tf.layers.batch_normalization(conv2d_output,
                                           training=training)
        return leaky_relu(bn, name='outputs')


class Generator(object):
    def __init__(self, channels, init_conv_size):
        assert len(channels) > 1
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.compat.v1.variable_scope('generator', reuse=self._reuse):
            with tf.compat.v1.variable_scope('inputs'):
                fc = tf.layers.dense(
                    inputs,
                    self._channels[0] * self._init_conv_size * self._init_conv_size)
                conv0 = tf.reshape(fc, [-1, self._init_conv_size, self._init_conv_size, self._channels[0]])
                bn0 = tf.layers.batch_normalization(conv0, training=training)
                relu0 = tf.nn.relu(bn0)

            deconv_inputs = relu0
            for i in range(1, len(self._channels)):
                with_bn_relu = (i != len(self._channels) - 1)
                deconv_inputs = conv2d_transpose(deconv_inputs,
                                                 self._channels[i],
                                                 'deconv-%d' % i,
                                                 training,
                                                 with_bn_relu)
            user_inputs = deconv_inputs
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return user_inputs


class Discriminator(object):
    def __init__(self, channels):
        self._channels = channels
        self._reuse = False

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        conv_inputs = inputs
        with tf.variable_scope('discriminator', reuse=self._reuse):
            for i in range(len(self._channels)):
                conv_inputs = conv2d(conv_inputs,
                                     self._channels[i],
                                     'deconv-%d' % i,
                                     training)
            fc_inputs = conv_inputs
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten, 2, name="logits")
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits


class DCGAN(object):
    def __init__(self, hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels
        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._batch_size = hps.batch_size
        self._z_dim = hps.z_dim
        self._fake_user_size = hps.fake_user_size

        self._generator = Generator(g_channels, self._init_conv_size)
        self._discriminator = Discriminator(d_channels)

    def build(self):
        tf.disable_eager_execution()
        self._z_placholder = tf.placeholder(tf.float32, (None, self._z_dim), name='z_placeholder')
        self._user_placeholder = tf.placeholder(tf.float32,
                                               (None, self._fake_user_size, self._fake_user_size, 1),name='user_placeholder')

        generated_users = self._generator(self._z_placholder, training=True)
        tf.identity(generated_users, name="g_user")
        fake_user_logits = self._discriminator(generated_users, training=True)
        real_user_logits = self._discriminator(self._user_placeholder, training=True)
        tf.identity(real_user_logits, name="detect")


        loss_on_fake_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size], dtype=tf.int64),
                logits=fake_user_logits))
        loss_on_fake_to_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self._batch_size], dtype=tf.int64),
                logits=fake_user_logits))
        loss_on_real_to_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size], dtype=tf.int64),
                logits=real_user_logits))

        tf.add_to_collection('g_losses', loss_on_fake_to_real)
        tf.add_to_collection('d_losses', loss_on_fake_to_fake)
        tf.add_to_collection('d_losses', loss_on_real_to_real)

        loss = {
            'g': tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            'd': tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }

        return (self._z_placholder, self._user_placeholder, generated_users, loss)

    def build_train(self, losses, learning_rate, beta1):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses['g'], var_list=self._generator.variables)
        d_opt_op = d_opt.minimize(losses['d'], var_list=self._discriminator.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')


def save(saver, sess, logdir, step1):
    model_name = 'model'
    checkpoint_path = os.path.join(logdir, model_name)
    saver.save(sess, checkpoint_path, global_step=step1)
    print('The checkpoint has been created.')

def DCGAN_User(Fake_User, cfg):
    output_dir = './local_run'
    if not os.path.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    if not os.path.exists('./models/'):
        tf.io.gfile.makedirs('./models/')

    hps = get_default_params(len(Fake_User), cfg)

    user_data = UserData(Fake_User, hps.z_dim, hps.fake_user_size)

    dcgan = DCGAN(hps)
    z_placeholder, user_placeholder, generated_users, losses = dcgan.build()
    saver = tf.compat.v1.train.Saver(var_list=dcgan._generator.variables, max_to_keep=1)  # 模型的保存器
    train_op = dcgan.build_train(losses, hps.learning_rate, hps.beta1)

    init_op = tf.global_variables_initializer()
    train_steps = cfg.epoch

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(train_steps):
            batch_user, batch_z = user_data.next_batch()

            fetches = [train_op, losses['g'], losses['d']]
            should_sample = (step + 1) % 10 == 0
            if should_sample:
                fetches += [generated_users]
            out_values = sess.run(fetches,
                                  feed_dict={
                                      z_placeholder: batch_z,
                                      user_placeholder: batch_user
                                  })
            _, g_loss_val, d_loss_val = out_values[0:3]
            tf.compat.v1.logging.info('step: %d, g_loss: %4.3f, d_loss: %4.3f' % (step, g_loss_val, d_loss_val))
            print('step: %d, g_loss: %4.3f, d_loss: %4.3f' % (step, g_loss_val, d_loss_val))
        save(saver, sess, './models/', step)

        out_values_final = sess.run(fetches,
                              feed_dict={
                                  z_placeholder: batch_z,
                                  user_placeholder: batch_user
                              })
        gen_users = out_values_final[3]
        gen_user_path = os.path.join(output_dir, '%05d-gen.txt' % (step + 1))
        gen_user = gen_users.reshape((gen_users.shape[0]), hps.fake_user_size * hps.fake_user_size)
        with open(gen_user_path, 'a') as f:
            for s in range(len(gen_user)):
                for i in range(len(gen_user[s])):
                    f.write(str(gen_user[s][i]) + " ")
                f.write("\n")

    return gen_user





