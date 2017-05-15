import tensorflow as tf
from tf_alexnet.alexnet import AlexNet
from filehandling_utils import load_image
import time
import numpy as np
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt


PARAMS = dict(image_path='./data/selected/images_resized/jacket.bmp', layer_name='conv1/relu:0',
              alpha=6, beta=2,
              lambda_a=2.16, lambda_b=0.5,
              sigma=1,
              learning_rate=0.1,
              num_iterations=4000,
              print_freq=100, log_freq=2000,
              log_path='./mahendran_vedaldi/run1/')


def alpha_norm_prior(tensor, alpha):
    return tf.norm(tensor - tf.reduce_mean(tensor), ord=alpha)


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
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            img_mat = load_image(params['image_path'], resize=False)
            image = tf.constant(img_mat, dtype=tf.float32, shape=[1, 224, 224, 3])
            reconstruction = tf.get_variable('reconstruction', shape=[1, 224, 224, 3], dtype=tf.float32)
            net_input = tf.concat([image, reconstruction], axis=0)

            alexnet = AlexNet()
            alexnet.build(net_input, rescale=1.0)

            representation = graph.get_tensor_by_name(params['layer_name'])

            img_rep = tf.slice(representation, [0, 0, 0, 0], [1, -1, -1, -1])
            rec_rep = tf.slice(representation, [1, 0, 0, 0], [1, -1, -1, -1])

            mse_loss = tf.losses.mean_squared_error(img_rep, rec_rep)
            prior_a = params['lambda_a'] * alpha_norm_prior(reconstruction, params['alpha'])
            prior_b = params['lambda_b'] * total_variation_prior(reconstruction, params['beta'])
            loss = mse_loss + prior_a + prior_b

            adam = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

            train_op = adam.minimize(loss)

            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            for count in range(params['num_iterations']):
                batch_loss, _, rec_mat = sess.run([loss, train_op, reconstruction])

                if (count + 1) % params['print_freq'] == 0:
                    print(('Iteration: {0:6d} Training Error:   {1:9.2f} ' +
                           'Time: {2:5.1f} min').format(count + 1, batch_loss, (time.time() - start_time) / 60))

                if (count + 1) % params['log_freq'] == 0:
                    plot_mat = np.zeros(shape=(224, 2 * 224, 3))

                    plot_mat[:, :224, :] = img_mat / 255.0
                    plot_mat[:, 224:, :] = np.minimum(np.maximum(rec_mat / 255.0, 0.0), 1.0)
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(2, 1)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(plot_mat, aspect='auto')
                    plt.savefig(params['log_path'] + 'rec_' + str(count + 1) + '.png', format='png', dpi=224)


invert_layer(PARAMS)