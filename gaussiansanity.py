import tensorflow as tf
import numpy as np


def gauss_J(x_mat, mu_vec, sig_inv_mat):
    print(sig_inv_mat)
    const_T = x_mat.get_shape()[0].value
    term1 = - tf.reduce_sum(tf.diag_part(sig_inv_mat), name='t1')
    # mat = tf.matmul(diff, sig_inv_mat)
    # term2 = 0.5 * tf.reduce_sum(mat**2)
    term2 = 0.5 * tf.reduce_sum(tf.matmul((x_mat - mu_vec), sig_inv_mat)**2, name='t2')
    return (term1 + term2) / const_T


def train():

    m_feats = 5
    n_comps = 20
    batch_size = 1000
    num_iterations = 100000
    lr = 1.0e-1
    grad_clip = 100.0
    true_mu = [1, 2, 3, 4, 5]
    true_sig = [0.1, 0.2, 0.3, 0.4, 0.5]

    x_mat = np.zeros((batch_size, m_feats))
    for idx, m, s in zip(range(m_feats), true_mu, true_sig):
        x_mat[:, idx] = np.random.normal(loc=m, scale=s, size=[batch_size])

    print(x_mat[:10, :])

    with tf.Graph().as_default() as graph:
        x_pl = tf.placeholder(dtype=tf.float32, shape=[batch_size, m_feats])
        lr_pl = tf.placeholder(dtype=tf.float32, shape=[])
        mu_vec = tf.get_variable(shape=[m_feats], dtype=tf.float32, name='mu',
                                 initializer=tf.random_normal_initializer())
        sig_inv_mat = tf.get_variable(shape=[m_feats, m_feats], dtype=tf.float32, name='siginv',
                                      initializer=tf.random_normal_initializer(stddev=1.0))

        loss = gauss_J(x_pl, mu_vec, sig_inv_mat)

        # opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        # opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=lr_pl)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        tg_pairs = [k for k in zip(grads, tvars) if k[0] is not None]
        # tg_clipped = [(tf.clip_by_value(k[0], -grad_clip, grad_clip), k[1])
        #               for k in tg_pairs]
        # opt_op = opt.apply_gradients(tg_clipped)
        opt_op = opt.apply_gradients(tg_pairs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for idx in range(num_iterations):

                batch_loss, _ = sess.run(fetches=[loss, opt_op], feed_dict={x_pl: x_mat, lr_pl: lr})
                # sess.run(clip_op)
                if idx % 1000 == 0:
                    print(batch_loss)
                    term1 = graph.get_tensor_by_name('t1:0')
                    term2 = graph.get_tensor_by_name('t2:0')
                    mu, sig, t1, t2 = sess.run([mu_vec, sig_inv_mat, term1, term2], feed_dict={x_pl: x_mat})
                    # print(mu)
                    # print(np.linalg.pinv(sig))
                    print(t1, t2)

                if idx % 10000 == 0:
                    print(mu)
                    print(np.linalg.pinv(sig))

                if (idx + 1) % 20000 == 0:
                    lr /= 10
                    print('new lr:' + str(lr))

train()