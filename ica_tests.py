import tensorflow as tf
import numpy as np
import filehandling_utils
from skimage.color import rgb2gray
import matplotlib
matplotlib.use('tkagg', force=True)
import matplotlib.pyplot as plt

def cosh(x):
    return 0.5 * tf.exp(x, name='exp_x') + 0.5 * tf.exp(-x, name='exp_negx')


def func_G(s):
    a = np.pi * s / (2 * 3**0.5)
    return -2 * tf.log(cosh(a)) - tf.log(4)


def func_g(s):
    a = np.pi * s / (2 * 3**0.5)
    return - np.pi / 3 * tf.tanh(a)


def func_g_prime(s):
    a = np.pi * s / (3 ** 0.5)
    b = 3 * 3**0.5 * (cosh(a) + 1)
    b = tf.identity(b, name='g_prime_b')
    return - np.pi**2 / b


def func_phi(x, w_mat):
    g_vec = func_g(tf.matmul(w_mat, x, transpose_a=True))
    return tf.matmul(w_mat, g_vec)


def func_partial_phi_i():
    pass


def loss_J_tilde(x_mat, w_mat, alpha):
    const_T = x_mat.get_shape()[0].value
    xw_mat = tf.matmul(x_mat, w_mat)
    g_mat = func_g(xw_mat)
    gp_mat = func_g_prime(xw_mat)
    gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_T
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
    aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
    ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)

    term_1 = tf.reduce_sum(alpha * gp_vec)
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat)
    return term_1 + term_2


def old_loss_J_tilde(x_mat, w_mat):
    const_T = x_mat.get_shape()[0].value
    xw_mat = tf.matmul(x_mat, w_mat, name='xw_mat')
    g_mat = func_g(xw_mat)
    gp_mat = func_g_prime(xw_mat)
    gp_vec = tf.reduce_sum(gp_mat, axis=0, name='gp_vec') / const_T
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
    ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
    w_norm = tf.diag_part(ww_mat, name='w_norm')
    term_1 = tf.reduce_sum(w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(ww_mat * gg_mat, name='t2')
    return term_1 + term_2


def neg_log_p(x_mat, w_mat):
    return - tf.reduce_sum(func_G(x_mat @ w_mat), axis=1)

### TRAIN UTILS ###


def get_batch_gen(batch_size, patch_size=(8, 8)):
    img_hw = 224
    max_h = img_hw - patch_size[0]
    max_w = img_hw - patch_size[1]
    data_path = './data/imagenet2012-validationset/'
    img_file = 'train_48k_images.txt'
    begin = 0

    with open(data_path + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    while True:

        end = begin + batch_size
        if end < len(image_files):
            batch_files = image_files[begin:end]
        else:
            end = end - len(image_files)
            batch_files = image_files[begin:] + image_files[:end]
        begin = end
        batch_paths = [data_path + 'images_resized/' +
                       k[:-len('JPEG')] + 'bmp' for k in batch_files]
        images = []
        for img_path in batch_paths:
            image = filehandling_utils.load_image(img_path, resize=False)
            image = rgb2gray(image)
            image /= np.max(image)
            image -= np.mean(image)
            images.append(image)
        mat = np.stack(images, axis=0)

        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        # yield mat[:, h:h + patch_size[0], w:w + patch_size[1], :]
        yield mat[:, h:h + patch_size[0], w:w + patch_size[1]]


def train_test():
    ph = 8
    pw = 8
    m_feats = ph * pw
    n_comps = m_feats
    # assert m_feats <= n_comps
    batch_size = 5000
    num_iterations = 10000
    lr = 1.0e-4
    grad_clip = 10.0

    bgen = get_batch_gen(batch_size, patch_size=(ph, pw))
    with tf.Graph().as_default() as graph:

        img_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, ph, pw])
        w_mat = tf.get_variable(shape=[m_feats, n_comps], dtype=tf.float32, name='W',
                                initializer=tf.random_normal_initializer())
        # alpha = tf.get_variable(shape=[n_comps, 1], dtype=tf.float32, name='alpha',
        #                         initializer=tf.constant_initializer(value=1.5))

        x_mat = tf.reshape(img_pl, [batch_size, m_feats], name='X')
        loss = old_loss_J_tilde(x_mat=x_mat, w_mat=w_mat)

        clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat))

        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
        # tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
        #               for k in tg_pairs]
        # opt_op = opt.apply_gradients(tg_clipped)
        opt_op = opt.apply_gradients(tg_pairs)
        # opt_op = opt.minimize(loss)
        # loss = tf.Print(loss, tf.gradients(loss, w_mat)) nan
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('t1:0'), w_mat), message='t1:') nan
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('t2:0'), w_mat), message='t2:') is fine
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('w_norm:0'), w_mat), message='w_norm:') is fine
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('gp_vec:0'), w_mat), message='gp_vec:')
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('exp_negx:0'), w_mat), message='exp_negx:')
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('exp_x:0'), w_mat), message='exp_x:')
        # loss = tf.Print(loss, tf.gradients(graph.get_tensor_by_name('xw_mat:0'), w_mat), message='xw_mat:')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            patches = next(bgen)
            patches = patches.astype(float)
            # patches /= 255.0
            # patches -= np.mean(patches)
            # sess.run(clip_op)
            for idx in range(num_iterations):
                term1 = graph.get_tensor_by_name('w_norm:0')
                term2 = graph.get_tensor_by_name('gp_vec:0')
                batch_loss, t1, t2, _ = sess.run(fetches=[loss, term1, term2, opt_op], feed_dict={img_pl: patches})

                if idx % 100 == 0:
                    print(batch_loss)

            w_res = sess.run(w_mat)
            p1 = patches[0, :, :]
            plt.figure()
            plt.imshow(p1, cmap='gray')
            for idx in range(4):
                plt.figure()
                f1 = np.reshape(w_res[:, idx], [ph, pw])
                plt.imshow(f1, cmap='gray')
            plt.show()

train_test()

