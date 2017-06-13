import tensorflow as tf
import numpy as np
import filehandling_utils

def cosh(x):
    return 0.5 * tf.exp(x) + 0.5 * tf.exp(-x)


def func_G(s):
    a = np.pi * s / (2 * 3**0.5)
    return -2 * tf.log(cosh(a)) - tf.log(4)


def func_g(s):
    a = np.pi * s / (2 * 3**0.5)
    return - np.pi / 3 * tf.tanh(a)


def func_g_prime(s):
    a = np.pi * s / (3 ** 0.5)
    b = 3 * 3**0.5 * (cosh(a) + 1)
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


def neg_log_p(x_mat, w_mat):
    return - tf.reduce_sum(func_G(x_mat @ w_mat), axis=1)


### TRAIN UTILS ###

def get_batch_gen(batch_size, patch_size=(8,8)):
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
            images.append(image)
        mat = np.stack(images, axis=0)

        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        yield mat[:, h:h + patch_size[0], w:w + patch_size[1], :]


def train_test():
    ph = 5
    pw = 5
    n_comps = 100
    m_feats = ph * pw * 3
    batch_size = 100
    num_iterations = 100
    lr = 0.000001
    bgen = get_batch_gen(batch_size, patch_size=(ph, pw))
    with tf.Graph().as_default() as graph:

        img_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, ph, pw, 3])
        w_mat = tf.get_variable(shape=[m_feats, n_comps], dtype=tf.float32, name='W',
                                initializer=tf.random_normal_initializer())
        alpha = tf.get_variable(shape=[n_comps, 1], dtype=tf.float32, name='alpha',
                                initializer=tf.constant_initializer(value=1.5))

        x_mat = tf.reshape(img_pl, [batch_size, m_feats], name='X')
        loss = loss_J_tilde(x_mat=x_mat, w_mat=w_mat, alpha=alpha)

        clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat))

        opt = tf.train.AdamOptimizer(learning_rate=lr)
        opt_op = opt.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for idx in range(num_iterations):
                patches = next(bgen)

                batch_loss, _ = sess.run(fetches=[loss, opt_op], feed_dict={img_pl: patches})

                if idx % 1 == 0:
                    print(batch_loss)

train_test()

