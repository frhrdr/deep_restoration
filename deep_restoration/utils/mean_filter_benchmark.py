import foolbox
import numpy as np
import tensorflow as tf

from utils.filehandling import load_image
from utils.foolbox_utils import advex_match_paths, get_classifier_io, get_attack
from utils.imagenet_classnames import get_class_name


def default_mean_filter_exp():
    weighings = (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
    mean_filter_benchmark(classifier='alexnet', filter_hw=2, weightings=weighings)
    mean_log_statistics(plot=False)


def mean_filter_benchmark(classifier, filter_hw, weightings):
    """
    evaluates weighted mean filter performances on oblivious adversarials
    """
    log_list = []

    # advex_matches = advex_match_paths(images_file='alexnet_val_2k_top1_correct.txt',
    #                                   advex_subdir='alexnet_val_2k_top1_correct/deepfool_oblivious/')
    advex_matches = advex_match_paths(images_file='subset_100_images.txt',
                                      advex_subdir='100_dataset/deepfool_oblivious/')

    print('number of matches:', len(advex_matches))
    count = 0

    with tf.Graph().as_default():

        smoothed_img, img_pl, mean_filter_pl, filter_feed_op = mean_filter_model(filter_hw)
        _, logit_tsr = get_classifier_io(classifier, input_init=smoothed_img, input_type='tensor')
        ref_in, ref_out = get_classifier_io(classifier, input_type='placeholder')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for img_path, adv_path in advex_matches[:2]:
                print(img_path, adv_path)
                count += 1
                print('match no.', count)
                image = np.expand_dims(load_image(img_path).astype(dtype=np.float32), axis=0)
                advex = np.expand_dims(load_image(adv_path).astype(dtype=np.float32), axis=0)
                print(np.linalg.norm(image - advex))

                ref_img = sess.run(ref_out, feed_dict={ref_in: image})
                ref_adv = sess.run(ref_out, feed_dict={ref_in: advex})

                print(np.argmax(ref_img), np.argmax(ref_adv))
                img_log_list = []
                adv_log_list = []
                for weight in weightings:
                    filter_mat = make_weighted_mean_filter(weight, filter_hw)
                    sess.run(filter_feed_op, feed_dict={mean_filter_pl: filter_mat})

                    img_smoothed_pred, img_smoothed = sess.run([logit_tsr, smoothed_img], feed_dict={img_pl: image})
                    img_smoothed_label = np.argmax(img_smoothed_pred)
                    img_log_list.append(img_smoothed_label)

                    adv_smoothed_pred, adv_smoothed = sess.run([logit_tsr, smoothed_img], feed_dict={img_pl: advex})
                    adv_smoothed_label = np.argmax(adv_smoothed_pred)
                    adv_log_list.append(adv_smoothed_label)
                    print(weight, np.max(filter_mat), 'norms', np.linalg.norm(advex - adv_smoothed),
                          np.linalg.norm(image - img_smoothed))
                    # print((image - img_smoothed)[0, :10, :10, 0])
                    # print((advex - adv_smoothed)[0, :10, :10, 0])
                    print(image[0, :10, :10, 0])
                    print(img_smoothed[0, :10, :10, 0])
                    print(filter_mat[:, :, 0, :])
                log_list.append([img_log_list, adv_log_list])
    log_mat = np.asarray(log_list)
    print(log_mat)
    np.save('smooth_log.npy', log_mat)


def mean_log_statistics(log_path='smooth_log.npy', plot=True, weightings=None):
    weightings = weightings or (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
    log = np.load(log_path)
    print('log shape', log.shape)
    print('number of samples contained:', log.shape[0])
    orgiginal_labels = np.stack([log[:, 0, 0]] * log.shape[2], axis=1)
    a = np.sum(orgiginal_labels == log[:, 0, :], axis=0)
    print('labels retained after smoothing image', a)
    a = np.sum(orgiginal_labels == log[:, 1, :], axis=0)
    print('labels restored after smoothing advex', a)
    a = np.sum(log[:, 0, :] == log[:, 1, :], axis=0)
    print('image and advex smoothed to same label', a)


def mean_adaptive_attacks_200(attack_name='deepfool', attack_keys=None, verbose=True):
    """
    creates adaptive adversarials for mean filter defense
    :param attack_name:
    :param attack_keys:
    :param verbose:
    :return:
    """
    path = '../logs/adversarial_examples/deepfool_oblivious_198/'
    img_log = np.load(path + 'img_log_198_fine.npy')
    # adv_log = np.load(path + 'adv_log_198_fine.npy')
    classifier = 'alexnet'
    advex_matches = advex_match_paths(images_file='subset_cutoff_200_images.txt',
                                      advex_subdir='200_dataset/deepfool_oblivious/')

    with tf.Graph().as_default():

        net_input, img_pl, _ = mean_filter_model(make_switch=False)
        input_var, logit_tsr = get_classifier_io(classifier, input_init=net_input, input_type='tensor')

        with tf.Session() as sess:
            model = foolbox.models.TensorFlowModel(img_pl, logit_tsr, bounds=(0, 255))
            criterion = foolbox.criteria.Misclassification()
            attack = get_attack(attack_name, model, criterion)
            if attack_keys is None:
                attack_keys = dict()

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            noise_norms = []
            src_invariant = []
            for idx, match in enumerate(advex_matches[:20]):
                img_path, adv_path = match
                src_label = img_log[idx][0]

                img = load_image(img_path)
                adv = load_image(adv_path)

                oblivious_norm = np.linalg.norm((img - adv).flatten(), ord=2)
                print('oblivious norm', oblivious_norm)

                img_pred = model.predictions(img)
                img_pred_label = np.argmax(img_pred)
                adv_pred = model.predictions(adv)
                adv_pred_label = np.argmax(adv_pred)

                if verbose:
                    noisy_label_name = get_class_name(img_pred_label)
                    if img_pred_label == src_label:
                        print('noisy label {} same as source: {}'.format(img_pred_label, noisy_label_name))
                        src_invariant.append(1)
                        if adv_pred_label == img_pred_label:
                            print('oblivious attack averted')
                        else:

                            print('WARNING: oblivious attack succeeded!')
                    else:
                        print('image with prior misclassified as {}. (label {})'.format(noisy_label_name,
                                                                                        img_pred_label))
                        src_invariant.append(0)
                adversarial = attack(image=img, label=src_label, **attack_keys)
                if adversarial is None:
                    adaptive_norm = None
                    if verbose:
                        print('no adversary found for source label {} using {}'.format(img_pred_label, attack_name))
                else:
                    fooled_pred = model.predictions(adversarial)
                    fooled_label = np.argmax(fooled_pred)
                    fooled_label_name = get_class_name(fooled_label)
                    adaptive_norm = np.linalg.norm((img - adversarial).flatten(), ord=2)
                    if verbose:
                        print('adversarial image classified as {}. (label {}) '
                              'Necessary perturbation: {}'.format(fooled_label_name, fooled_label, adaptive_norm))
                    # adaptive_save_path = adv_path.replace('oblivious', 'adaptive'
                    # np.save(adaptive_save_path, adversarial)
                noise_norms.append((oblivious_norm, adaptive_norm))


def mean_filter_model(filter_hw):
    """
    makes a mean-smoothed model with a feedable filter
    :param filter_hw:
    :return:
    """
    pad_u = int(np.ceil((filter_hw - 1) / 2))
    pad_d = int(np.floor((filter_hw - 1) / 2))

    image_shape = (1, 227, 227, 3)
    mean_filter_pl = tf.placeholder(dtype=tf.float32, shape=(filter_hw, filter_hw, 3, 1), name='filter_pl')
    mean_filter_var = tf.get_variable('filter_var', shape=(filter_hw, filter_hw, 3, 1),
                                      dtype=tf.float32, trainable=False)
    img_pl = tf.placeholder(dtype=tf.float32, shape=image_shape)
    img_tsr = tf.pad(img_pl, paddings=[(0, 0), (pad_u, pad_d), (pad_u, pad_d), (0, 0)], mode='REFLECT')
    smoothed_img = tf.nn.depthwise_conv2d(img_tsr, mean_filter_var, strides=[1, 1, 1, 1], padding='VALID')

    filter_feed_op = tf.assign(mean_filter_var, mean_filter_pl)
    return smoothed_img, img_pl, mean_filter_pl, filter_feed_op


def make_weighted_mean_filter(smoothness_weight, filter_hw):
    """
    makes a weighted mean filter numpy mat
    :param smoothness_weight:
    :param filter_hw:
    :return:
    """
    center_weight = smoothness_weight / filter_hw ** 2 + 1 - smoothness_weight
    off_center_weight = smoothness_weight / filter_hw ** 2
    center_idx_hw = (filter_hw - 1) // 2
    mean_filter_mat = np.ones(shape=[filter_hw, filter_hw, 3, 1]) * off_center_weight
    mean_filter_mat[center_idx_hw, center_idx_hw, :, :] = center_weight
    return mean_filter_mat
