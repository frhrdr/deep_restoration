import tensorflow as tf
import numpy as np
import filehandling_utils
from skimage.color import rgb2gray
from sklearn.decomposition import PCA, FastICA
from sklearn.datasets import fetch_olivetti_faces
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
    return - np.pi * tf.tanh(a) / 3**0.5


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
    g_mat = -tf.tanh(xw_mat)
    gp_mat = -4.0 / tf.square(tf.exp(xw_mat) + tf.exp(-xw_mat))  # d/dx tanh(x) = 4 / (exp(x) + exp(-x))^2
    gp_vec = tf.reduce_sum(gp_mat, axis=0) / const_T
    gg_mat = tf.matmul(g_mat, g_mat, transpose_a=True) / const_T
    aa_mat = tf.matmul(alpha, alpha, transpose_b=True)
    ww_mat = tf.matmul(w_mat, w_mat, transpose_a=True)
    w_norm = tf.diag_part(ww_mat, name='w_norm')

    term_1 = tf.reduce_sum(alpha * w_norm * gp_vec, name='t1')
    term_2 = 0.5 * tf.reduce_sum(aa_mat * ww_mat * gg_mat, name='t2')
    return term_1 + term_2


def old_loss_J_tilde(x_mat, w_mat, alpha=None):
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

def plot_img_mats(mat, color=False):
    """ plot l*m*n mats as l m by n gray-scale images """
    n = mat.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n // cols))
    mat = np.maximum(mat, 0.0)
    mat = np.minimum(mat, 1.0)
    if not color:
        plt.style.use('grayscale')
    fig, ax_list = plt.subplots(ncols=cols, nrows=rows)
    ax_list = ax_list.flatten()

    for idx, ax in enumerate(ax_list):
        if idx >= n:
            ax.axis('off')
        else:
            if color:
                ax.imshow(mat[idx, :, :, :], interpolation='none')
            else:
                ax.imshow(mat[idx, :, :], interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


def pca_whiten_as_ica(data):
    n_samples, n_features = data.shape
    data = data.T
    data -= data.mean(axis=-1)[:, np.newaxis]
    u, d, _ = np.linalg.svd(data, full_matrices=False)
    K = (u / d).T[:n_features]
    data = np.dot(K, data)
    data *= np.sqrt(n_samples)
    return data.T, K


def pca_whiten_as_pca(data):
    n_samples, n_features = data.shape
    u, s, v = np.linalg.svd(data, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    u *= np.sqrt(n_samples)
    v *= signs[:, np.newaxis]
    rerotate = v / s[:, np.newaxis] * np.sqrt(n_samples)
    return u, rerotate


def zca_whiten(data):
    cov = np.dot(data.T, data) / (data.shape[0] - 1)
    U, S, _ = np.linalg.svd(cov)
    s = np.sqrt(S.clip(1.e-6))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whiten = np.dot(np.dot(U, s_inv), U.T)
    dewhiten = np.dot(np.dot(U, s), U.T)
    data = np.dot(data, whiten.T)
    return data, dewhiten


def get_patches(num=1000, ph=8, pw=8, color=False):
    img_hw = 224
    max_h = img_hw - ph
    max_w = img_hw - pw
    data_path = './data/imagenet2012-validationset/'
    img_file = 'train_48k_images.txt'

    with open(data_path + img_file) as f:
        image_files = [k.rstrip() for k in f.readlines()]

    image_paths = [data_path + 'images_resized/' +
                   k[:-len('JPEG')] + 'bmp' for k in image_files[:num]]
    images = []

    for img_path in image_paths:
        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        image = filehandling_utils.load_image(img_path, resize=False)
        image = image.astype(float)
        image = image[h:h + ph, w:w + pw, :]
        if not color:
            image = rgb2gray(image)
        image = image.flatten()
        images.append(image)
    mat = np.stack(images, axis=0)
    mat /= 255.0  # map to range [0,1]
    mat_centered = mat - mat.mean(axis=1).reshape(num, -1)  # subtract image mean
    # don't subtract channel mean (stationarity should take care of this)
    # mat_centered = mat_centered - mat_centered.mean(axis=0)
    # print(mat_centered.mean(axis=0))
    # print(mat_centered.mean(axis=1))
    return mat_centered


def get_faces():
    faces = fetch_olivetti_faces(shuffle=True).data
    n_samples, n_features = faces.shape
    faces_centered = faces - faces.mean(axis=0)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    return faces_centered


def train_test():
    ph = 8
    pw = 8
    color = True
    n_features = ph * pw
    if color:
        n_features *= 3
    n_components = 128
    n_samples = 1000
    num_iterations = 50000
    lr = 3.0e-7
    lr_lower_points = [(0, 1.0e-5), (40000, 3.0e-6), (50000, 1.0e-6), (60000, 1.0e-6), (70000, 3.0e-7)]
    grad_clip = 100.0
    n_vis = 100
    sgd_based = True

    data = get_patches(n_samples, ph, pw, color=color)
    data, rerotate = pca_whiten_as_ica(data)
    # pca = PCA(n_components=n_features, whiten=True)
    # data = pca.fit_transform(data)

    with tf.Graph().as_default() as graph:

        x_pl = tf.placeholder(dtype=tf.float32, shape=[n_samples, n_features], name='x_pl')
        w_mat = tf.get_variable(shape=[n_features, n_components], dtype=tf.float32, name='w_mat',
                                initializer=tf.random_normal_initializer(stddev=0.001))
        alpha = tf.get_variable(shape=[n_components, 1], dtype=tf.float32, name='alpha',
                                initializer=tf.random_normal_initializer())
        lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

        loss = loss_J_tilde(x_mat=x_pl, w_mat=w_mat, alpha=alpha)
        clip_op = tf.assign(w_mat, w_mat / tf.norm(w_mat, ord=2, axis=0))

        if sgd_based:
            opt = tf.train.MomentumOptimizer(learning_rate=lr_pl, momentum=0.9)
            # opt = tf.train.AdamOptimizer(learning_rate=lr_pl)  # , epsilon=0.001)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
            tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
                          for k in tg_pairs]
            opt_op = opt.apply_gradients(tg_clipped)
            opt_op = opt.apply_gradients(tg_pairs)
        else:
            # norm_constraints = tf.split(tf.norm(w_mat, ord=2, axis=0) - 1., num_or_size_splits=n_components)
            train_step = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                                options={'maxiter': num_iterations}
                                                                # , equalities=norm_constraints,
                                                                # inequalities=[tf.constant(1, dtype=tf.float32, shape=[1])]
                                                                )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(clip_op)
            if not sgd_based:
                num_iterations = 1

            for idx in range(num_iterations):
                if lr_lower_points and lr_lower_points[0][0] == idx:
                    lr = lr_lower_points[0][1]
                    print('new learning rate: ', lr)
                    lr_lower_points = lr_lower_points[1:]

                if sgd_based:
                    batch_loss, _ = sess.run(fetches=[loss, opt_op], feed_dict={x_pl: data, lr_pl: lr})
                    sess.run(clip_op)

                else:
                    batch_loss = sess.run(fetches=[loss], feed_dict={x_pl: data})
                    train_step.minimize(sess, feed_dict={x_pl: data})

                if idx % 100 == 0:
                    print(batch_loss)

                if idx % 1000 == 0:
                    term1 = graph.get_tensor_by_name('t1:0')
                    term2 = graph.get_tensor_by_name('t2:0')
                    w_res, alp, t1, t2 = sess.run([w_mat, alpha, term1, term2], feed_dict={x_pl: data})
                    print('it: ', idx, ' / ', num_iterations)
                    print('mean a: ', np.mean(alp), ' max a: ', np.max(alp), ' min a: ', np.min(alp))
                    print('mean w: ', np.mean(w_res), ' max w: ', np.max(w_res), ' min w: ', np.min(w_res))
                    print('term1: ', t1, ' term2: ', t2)

            w_res, alp = sess.run([w_mat, alpha])
            comps = np.dot(w_res.T, rerotate)
            print(comps.shape)
            comps -= np.min(comps)
            comps /= np.max(comps)
            if color:
                co = np.reshape(comps[:n_vis, :], [-1, ph, pw, 3])
            else:
                co = np.reshape(comps[:n_vis, :], [-1, ph, pw])
            plot_img_mats(co, color=color)


def fast_ica_comp():
    ph = 8
    pw = 8
    n_features = 64
    n_samples = 48000

    # data = get_patches(n_samples, ph, pw, color=True)
    # data, rerotate = pca_whiten_as_pca(data)
    # ica = FastICA(whiten=False, max_iter=1000)
    # ica.fit(data)
    # comps = ica.components_
    # comps = np.dot(comps, rerotate)
    # comps -= np.min(comps)
    # comps /= np.max(comps)
    # plot_img_mats(np.reshape(comps, [-1, ph, pw, 3]))

    data = get_patches(n_samples, ph, pw, color=True)
    print(data.shape)
    ica2 = FastICA(whiten=True, n_components=n_features, max_iter=1000)
    ica2.fit(data)
    comps = ica2.components_
    comps -= np.min(comps)
    comps /= np.max(comps)
    print(comps.shape)
    plot_img_mats(np.reshape(comps, [-1, ph, pw, 3]), color=True)

# train_test()
fast_ica_comp()


