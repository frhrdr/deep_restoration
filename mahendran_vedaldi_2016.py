import matplotlib
matplotlib.use('tkagg', force=True)
import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from tf_vgg.vgg16 import Vgg16
from filehandling_utils import load_image
import time
from filehandling_utils import save_dict
import numpy as np
import matplotlib.pyplot as plt
import os
from loss_modules import SoftRangeLoss, TotalVariationLoss, NormedMSELoss, LearnedPriorLoss
from ica_prior import ICAPrior

PARAMS = dict(image_path='./data/selected/images_resized/val13_monkey.bmp', layer_name='conv3/relu:0',
              classifier='alexnet',
              img_HW=224,
              jitter_T=8,
              mse_C=1.0,
              alpha_sr=6, beta_tv=2,
              range_B=80,
              range_V=80/6.5,
              learning_rate=0.04,
              num_iterations=10000,
              print_freq=50, log_freq=500, summary_freq=10, lr_lower_freq=500,
              grad_clip=100.0,
              log_path='./logs/mahendran_vedaldi/2016/',
              save_as_mat=False)


def invert_layer(params):
    if not os.path.exists(params['log_path']):
        os.makedirs(params['log_path'])

    save_dict(params, params['log_path'] + 'params.txt')

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            img_mat = load_image(params['image_path'], resize=False)
            image = tf.constant(img_mat, dtype=tf.float32, shape=[1, params['img_HW'], params['img_HW'], 3])
            rec_init = tf.abs(tf.random_normal([1, params['img_HW'], params['img_HW'], 3], mean=0, stddev=0.0001))
            reconstruction = tf.get_variable('reconstruction', dtype=tf.float32,
                                             initializer=rec_init)

            jitter_x_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_x_pl')
            jitter_y_pl = tf.placeholder(dtype=tf.int32, shape=[], name='jitter_y_pl')

            rec_part = tf.slice(reconstruction, [0, jitter_x_pl, jitter_y_pl, 0], [-1, -1, -1, -1])
            rec_padded = tf.pad(rec_part, paddings=[[0, 0], [jitter_x_pl, 0], [jitter_y_pl, 0], [0, 0]])

            net_input = tf.concat([image, rec_padded], axis=0)

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

            mse_mod = NormedMSELoss(target=img_rep.name, reconstruction=rec_rep.name, weighting=params['mse_C'])
            sr_mod = SoftRangeLoss(reconstruction.name, params['alpha_sr'],
                                   weighting=1 / (params['img_HW'] ** 2 * params['range_B'] ** params['alpha_sr']))
            tv_mod = TotalVariationLoss(reconstruction.name, params['beta_tv'],
                                        weighting=1 / (params['img_HW'] ** 2 * params['range_V'] ** params['beta_tv']))

            ica_prior = ICAPrior(tensor_names='reconstruction/read:0',
                                 weighting=0.0005, name='ICAPrior',
                                 load_path='./logs/priors/ica_prior/8by8_512_color/ckpt-5000',
                                 trainable=False, filter_dims=[8, 8], input_scaling=1.0, n_components=512)

            loss_mods = [mse_mod, ica_prior]  # [mse_mod, sr_mod, tv_mod]
            loss = 0
            for mod in loss_mods:
                mod.build()
                loss += mod.get_loss()

            lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
            # optimizer = tf.train.AdamOptimizer(lr_pl)
            optimizer = tf.train.MomentumOptimizer(lr_pl, momentum=0.9)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
            tg_clipped = [(tf.clip_by_value(k[0], -params['grad_clip'], params['grad_clip']), k[1])
                          for k in tg_pairs]
            train_op = optimizer.apply_gradients(tg_clipped)

            clip_op = tf.assign(reconstruction, tf.clip_by_value(reconstruction, 0.0, 2.0 * params['range_B']))

            for pair in tg_pairs:
                tf.summary.histogram(pair[0].name, pair[1])

            tf.summary.scalar('0_total_loss', loss)
            for mod in loss_mods:
                mod.scalar_summary()

            train_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(params['log_path'] + '/summaries', graph=graph)

            sess.run(tf.variables_initializer(tf.global_variables()))
            for mod in loss_mods:
                if isinstance(mod, LearnedPriorLoss):
                    mod.load_weights(sess)

            start_time = time.time()
            for count in range(params['num_iterations']):
                jitter = np.random.randint(low=0, high=params['jitter_T'] + 1, dtype=int, size=(2,))

                if (count + 1) % params['lr_lower_freq'] == 0:
                    params['learning_rate'] = params['learning_rate'] / 10.0
                    print('lowering learning rate to {0}'.format(params['learning_rate']))

                feed = {lr_pl: params['learning_rate'], jitter_x_pl: jitter[0], jitter_y_pl: jitter[1]}

                batch_loss, _, summary_string = sess.run([loss, train_op, train_summary_op], feed_dict=feed)
                # sess.run(check, feed_dict=feed)
                # sess.run(clip_op)
                rec_mat = sess.run(reconstruction)
                if (count + 1) % params['summary_freq'] == 0:
                    summary_writer.add_summary(summary_string, count)

                if (count + 1) % params['print_freq'] == 0:
                    print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                           'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))
                    # a = sess.run(graph.get_tensor_by_name('ICAPrior/ica_a:0'))
                    # print(a)
                    # w = sess.run(graph.get_tensor_by_name('ICAPrior/ica_w:0'))
                    # print(w)

                if (count + 1) % params['log_freq'] == 0:
                    plot_mat = np.zeros(shape=(params['img_HW'], 2 * params['img_HW'], 3))

                    plot_mat[:, :params['img_HW'], :] = img_mat / 255.0
                    rec_mat = (rec_mat - np.min(rec_mat)) / (np.max(rec_mat) - np.min(rec_mat))  # M&V just rescale
                    plot_mat[:, params['img_HW']:, :] = rec_mat
                    if params['save_as_mat']:
                        np.save(params['log_path'] + 'rec_' + str(count + 1) + '.npy', plot_mat)
                    else:
                        fig = plt.figure(frameon=False)
                        fig.set_size_inches(2, 1)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        ax.imshow(plot_mat, aspect='auto')
                        plt.savefig(params['log_path'] + 'rec_' + str(count + 1) + '.png',
                                    format='png', dpi=params['img_HW'])
