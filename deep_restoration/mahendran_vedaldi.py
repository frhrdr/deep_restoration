import matplotlib
matplotlib.use('tkagg', force=True)
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
from utils.filehandling import load_image, save_dict
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.ica_prior import ICAPrior
from modules.loss_modules import NormedMSELoss

PARAMS = dict(image_path='./data/selected/images_resized/val13_monkey.bmp', layer_name='conv3/relu:0',
              classifier='alexnet',
              alpha_l2=6, beta_tv=2,
              lambda_l2=2.16e+8, lambda_tv=5e+1,
              sigma=2.7098e+4,
              learning_rate=0.004,
              num_iterations=10000,
              print_freq=50, log_freq=500, summary_freq=10, lr_lower_freq=500,
              grad_clip=100.0,
              log_path='./logs/mahendran_vedaldi/run3/',
              save_as_mat=False)


def alpha_norm_prior(tensor, alpha):
    norm = tf.norm(tensor - tf.reduce_mean(tensor, axis=[0, 1, 2]), ord=alpha, axis=3)
    return tf.reduce_sum(norm ** alpha)


def total_variation_prior(tensor, beta):
    h0 = tf.slice(tensor, [0, 0, 0, 0], [-1, tensor.get_shape()[1].value - 1, -1, -1])
    h1 = tf.slice(tensor, [0, 1, 0, 0], [-1, -1, -1, -1])

    v0 = tf.slice(tensor, [0, 0, 0, 0], [-1, -1, tensor.get_shape()[1].value - 1, -1])
    v1 = tf.slice(tensor, [0, 0, 1, 0], [-1, -1, -1, -1])

    h_diff = h0 - h1
    v_diff = v0 - v1

    d_sum = tf.reduce_sum(h_diff * h_diff) + tf.reduce_sum(v_diff * v_diff)
    return d_sum ** (beta/2)


def invert_layer(params):
    if not os.path.exists(params['log_path']):
        os.makedirs(params['log_path'])

    save_dict(params, params['log_path'] + 'params.txt')

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            img_mat = load_image(params['image_path'], resize=False)
            image = tf.constant(img_mat, dtype=tf.float32, shape=[1, 224, 224, 3])
            rec_init = tf.abs(tf.random_normal([1, 224, 224, 3], mean=0, stddev=0.0001))
            reconstruction = tf.get_variable('reconstruction', dtype=tf.float32,
                                             initializer=rec_init)
            net_input = tf.concat([image, reconstruction * params['sigma']], axis=0)

            if params['classifier'].lower() == 'vgg16':
                classifier = Vgg16()
            elif params['classifier'].lower() == 'alexnet':
                classifier = AlexNet()
            else:
                raise NotImplementedError

            classifier.build(net_input, rescale=1.0)

            representation = graph.get_tensor_by_name(params['layer_name'])

            if len(representation.get_shape()) == 2:
                representation = tf.reshape(representation, shape=[2, -1, 1, 1])

            img_rep = tf.slice(representation, [0, 0, 0, 0], [1, -1, -1, -1])
            rec_rep = tf.slice(representation, [1, 0, 0, 0], [1, -1, -1, -1])
            mse_loss = NormedMSELoss(img_rep.name, rec_rep.name, weighting=1.0)
            # mse_loss = tf.reduce_sum((img_rep - rec_rep) ** 2) / tf.reduce_sum(img_rep * img_rep)
            prior_l2 = params['lambda_l2'] * alpha_norm_prior(reconstruction, params['alpha_l2'])
            prior_tv = params['lambda_tv'] * total_variation_prior(reconstruction, params['beta_tv'])
            # loss = mse_loss + prior_l2 + prior_tv

            ica_prior = ICAPrior(tensor_names='reconstruction/read:0',
                                 weighting=5.0e-6, name='ICAPrior',
                                 load_path='./logs/priors/ica_prior/color_8x8_512comps_191feats/ckpt-10000',
                                 trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512)

            ica_prior.build()
            mse_loss.build()
            loss = mse_loss.get_loss() + ica_prior.get_loss()

            lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
            # optimizer = tf.train.AdamOptimizer(lr_pl)
            optimizer = tf.train.MomentumOptimizer(lr_pl, momentum=0.9)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
            tg_clipped = [(tf.clip_by_value(k[0], -params['grad_clip'], params['grad_clip']), k[1])
                          for k in tg_pairs]
            train_op = optimizer.apply_gradients(tg_clipped)
            # train_op = optimizer.minimize(loss)
            for pair in tg_pairs:
                tf.summary.histogram(pair[0].name, pair[1])

            tf.summary.scalar('0_total_loss', loss)
            # tf.summary.scalar('1_mse_loss', mse_loss)
            tf.summary.scalar('2_l2_prior', prior_l2)
            tf.summary.scalar('3_tv_prior', prior_tv)

            train_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(params['log_path'] + '/summaries')

            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            for count in range(params['num_iterations']):
                if (count + 1) % params['lr_lower_freq'] == 0:
                    params['learning_rate'] = params['learning_rate'] / 10.0
                    print('lowering learning rate to {0}'.format(params['learning_rate']))

                batch_loss, _, rec_mat, summary_string = sess.run([loss, train_op, reconstruction, train_summary_op],
                                                                  feed_dict={lr_pl: params['learning_rate']})

                if (count + 1) % params['summary_freq'] == 0:
                    summary_writer.add_summary(summary_string, count)

                if (count + 1) % params['print_freq'] == 0:
                    print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                           'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))

                if (count + 1) % params['log_freq'] == 0:
                    plot_mat = np.zeros(shape=(224, 2 * 224, 3))

                    plot_mat[:, :224, :] = img_mat / 255.0

                    # rec_mat = np.minimum(np.maximum(rec_mat * params['sigma'] / 255.0, 0.0), 1.0)
                    rec_mat = (rec_mat - np.min(rec_mat)) / (np.max(rec_mat) - np.min(rec_mat))  # M&V just rescale
                    plot_mat[:, 224:, :] = rec_mat
                    if params['save_as_mat']:
                        np.save(params['log_path'] + 'rec_' + str(count + 1) + '.npy', plot_mat)
                    else:
                        fig = plt.figure(frameon=False)
                        fig.set_size_inches(2, 1)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        ax.imshow(plot_mat, aspect='auto')
                        plt.savefig(params['log_path'] + 'rec_' + str(count + 1) + '.png', format='png', dpi=224)
