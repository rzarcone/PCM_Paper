"""
Base code for fitting the memristor.
"""
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from utils import build_multilayer_network, memristor_output

def tensor_scaler(x, new_range):
    new_min, new_max = new_range
    # FIXME: depends on batch
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)

    return (((x-x_min)/(x_max-x_min))*(new_max-new_min)+new_min)

def get_snr(x, xh):
    return tf.reduce_mean(tf.reduce_sum(x**2,1)/tf.reduce_sum((x-xh)**2,1))


def _get_optim(optimizer_params):
    method = optimizer_params['method']
    if method == 'adam':
        learning_rate = optimizer_params['learning_rate']
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('Invalid Optimizer')

    return optim

def _train_graph(
    graph,
    init_op,
    train_op,
    cost_op, # FIXME unused
    summary_dict,
    saver,
    train_data_object,
    summary_data_object,
    feed_vars,
    batch_size,
    num_epochs,
    param_file):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(graph=graph, config=config) as sess:
        if param_file is None:
            sess.run(init_op)
        else:
            saver.restore(sess, param_file)
        for j in range(num_epochs):
            train_data_generator = train_data_object.get_generator(batch_size)
            for feed_values in train_data_generator:
                sess.run(
                    train_op,
                    feed_dict=dict(zip(feed_vars, feed_values))
                )
            summary_data_generator = summary_data_object.get_generator()
            for feed_values in summary_data_generator:
                summary_vals = sess.run(summary_dict,
                    feed_dict=dict(zip(feed_vars, feed_values))
                )
                tmp = ''
                for k, v in summary_vals.iteritems():
                    tmp += '{}: {} '.format(k, v)
                print(tmp)
        save_path = saver.save(sess, "tmp/model.ckpt")

    return save_path



def _eval_graph(
    graph,
    saver,
    eval_ops,
    data_object,
    feed_vars,
    param_file):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(graph=graph, config=config) as sess:
        saver.restore(sess, param_file)
        data_generator = data_object.get_generator()
        for feed_values in data_generator:
            eval_vals = sess.run(eval_ops,
                feed_dict=dict(zip(feed_vars, feed_values))
            )
    return eval_vals


class MemristorAutoEncoder(object):

    def __init__(
        self,
        data_dim,
        memristor_data={
            'vs_data': None,
            'mus_data': None,
            'sigs_data': None,
            'vmin': -1.,
            'vmax': 1.,
            'orig_r_range': (0, 1),
            'orig_v_range': (0, 1)
        },
        encoder_params={
            'n_layers': 3,
            'nonlinearity': tf.tanh},
        decoder_params={
            'n_layers': 3,
            'nonlinearity': tf.tanh
        },
        gamma=1.0,
        optimizer_params={
            'batch_size': 100,
            'num_epochs': 10,
            'method': 'adam',
            'learning_rate': 0.001
        },
        output_dir='output',
        param_file=None):
        """
        Initialize Memristor Autoencoder.

        Parameters
        ----------
        data_dim : int
            Dimension of the data.
        memristor_data : dict containing
            vs_data: array, shape (???
            'mus_data':None :
            'sigs_data':None :
            vmin: float
                Minimum voltage value after normalization
            vmax: float
                Maximum
            orig_r_range:
        FIXME: horrible o

        Returns
        -------

        """

        self.data_dim = data_dim
        self.memristor_data = memristor_data
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.gamma = gamma
        self.optimizer_params = optimizer_params
        self.output_dir = output_dir
        self.param_file = param_file

    def _build_graph(self):

        vs_data, mus_data, sigs_data = [self.memristor_data[key] for key in
                                        ['vs_data', 'mus_data', 'sigs_data']]
        # FIXME HACKY
        interp_width = np.array(vs_data[1, 0] - vs_data[0, 0]).astype('float32')

        n_samp, n_m = vs_data.shape

        g = tf.Graph()
        with g.as_default():

            x = tf.placeholder(tf.float32, shape=(None, self.data_dim)) # Input
            eps = tf.placeholder(tf.float32, shape=(None, n_m)) # Noise

            vmin, vmax = [tf.Variable(
                np.array(u).astype('float32'),
                trainable=False,
                name=name) for u, name in zip(
                    [self.memristor_data['vmin'], self.memristor_data['vmax']],
                    ['VMIN', 'VMAX'])]

            # Memristor Parameters
            vs, mus, sigs = [tf.Variable(
                initial_value=mat.astype('float32'),
                trainable=False,
                name=name)
                for mat, name in zip(
                    [vs_data, mus_data, sigs_data],
                    ['vs', 'mus', 'sigs'])]

            with tf.variable_scope('encoder'):
                v_raw = build_multilayer_network(
                    x,
                    self.encoder_params['layer_sizes'],
                    f=self.encoder_params['non_linearity'])
                v = tf.clip_by_value(v_raw, vmin, vmax)

            with tf.variable_scope('memristor'):
                r = memristor_output(v, eps, vs, mus, sigs, interp_width=interp_width)

            with tf.variable_scope('decoder'):
                xh = build_multilayer_network(
                    r,
                    self.decoder_params['layer_sizes'],
                    f=self.decoder_params['non_linearity'])


            with tf.variable_scope('costs'):
                # Penalty for going out of bounds
                gamma = tf.Variable(np.array(self.gamma).astype('float32'), trainable=False, name='GAMMA')
                reg_cost = tf.reduce_sum(gamma * (tf.nn.relu(v_raw - vmax) +
                                                  tf.nn.relu(vmin - v_raw))) # CHECK ME
                rec_cost = tf.nn.l2_loss(x - xh)  # Reconstruction cost
                cost = rec_cost + reg_cost


            with tf.variable_scope('visualize'):
                r_trans = tensor_scaler(r, self.memristor_data['orig_r_range'])
                v_trans = tensor_scaler(v, self.memristor_data['orig_v_range'])
                snr = get_snr(x, xh)

            train_op = _get_optim(self.optimizer_params).minimize(cost)

            saver = tf.train.Saver()

            init_op = tf.initialize_all_variables()




        d = {
            'graph': g,
            'cost': cost,
            'subcosts': {
                'reg_cost': reg_cost,
                'rec_cost': rec_cost
            },
            'feed_vars' : [x, eps],
            'xh': xh,
            'v': v_trans,
            'r': r_trans,
            'v_raw': v_raw,
            'init_op': init_op,
            'train_op': train_op,
            'saver': saver,
            'summary_ops': OrderedDict((
                ('cost', cost),
                ('rec_cost', rec_cost),
                ('reg_cost', reg_cost),
                ('snr', snr)
            ))
        }
        return d

    def fit(self, train_data_object, summary_data_object):
        d = self._build_graph()
        param_file = _train_graph(
            graph=d['graph'],
            init_op=d['init_op'],
            train_op=d['train_op'],
            cost_op=d['cost'],
            summary_dict=d['summary_ops'],
            saver=d['saver'],
            train_data_object=train_data_object,
            summary_data_object=summary_data_object,
            feed_vars=d['feed_vars'],
            batch_size=self.optimizer_params['batch_size'],
            num_epochs=self.optimizer_params['num_epochs'],
            param_file=self.param_file)
        self.param_file = param_file
        return self

    def score(self, X):
        return None

    def inspect_network(self, data_object):
        d = self._build_graph()

        eval_ops = {
            'x': d['feed_vars'][0],  # FIXME: Eww
            'xh': d['xh'],
            'v': d['v'],
            'r': d['r'],
            'v_raw': d['v_raw']
        }



        eval_vals = _eval_graph(
            graph=d['graph'],
            saver=d['saver'],
            eval_ops=eval_ops,
            data_object=data_object,
            feed_vars=d['feed_vars'],
            param_file=self.param_file
        )
        return eval_vals

