import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.client import device_lib
from model import VDCNN
devices = device_lib.list_local_devices()

parser = argparse.ArgumentParser(
    description="Test VDCNN model using test ARK files.")
parser.add_argument(
    "--TFRecord",
    type=str,
    default="output_TFRecord.tfrecords",
    help="Path and name of TFRecord file.")
parser.add_argument(
    "--model-path", type=str, default="model_save", help="Model save dir.")
parser.add_argument(
    "--model-name", type=str, default="saver-63", help="Saver name")
parser.add_argument(
    "--downsampling-type",
    type=str,
    default="maxpool",
    help=
    "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')"
)
args = parser.parse_args()
# Set some global variables
context_window_size = 11
feat_size = 43
batchsize = 32


def read_and_decode(TFRecord, context_window_size, feat_size):
    def _parse_record(example_proto):
        features = {
            "wav_feat": tf.FixedLenFeature([], tf.string),
            "noisy_feat": tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(
            example_proto, features=features)
        clean_feats = tf.decode_raw(parsed_features["wav_feat"], tf.float32)
        noise_feats = tf.decode_raw(parsed_features["noisy_feat"], tf.float32)

        sliced_feat = tf.reshape(clean_feats,
                                 [context_window_size, feat_size, 1])
        # noise_feats.set_shape([context_window_size * feat_size])
        sliced_noise_feat = tf.reshape(noise_feats,
                                       [context_window_size, feat_size, 1])
        return sliced_feat, sliced_noise_feat

    dataset = tf.data.TFRecordDataset(TFRecord)
    dataset = dataset.map(_parse_record).batch(batchsize)
    data_iterator = dataset.make_one_shot_iterator()
    sliced_feat, sliced_noise_feat = data_iterator.get_next()
    return sliced_feat, sliced_noise_feat


def clean(sess, model, sliced_noise_feat):
    c_res = None
    for beg_i in range(0, sliced_noise_feat.shape[1], context_window_size):
        if sliced_noise_feat.shape[1] - beg_i < context_window_size:
            length = sliced_noise_feat.shape[1] - beg_i
            pad = context_window_size - length
        else:
            length = context_window_size
            pad = 0
        x_ = np.zeros((batchsize, context_window_size))
        # if pad > 0:
        #     x_[0] = np.concatenate((sliced_noise_feat[beg_i:beg_i + length],
        #                             np.zeros(pad)))
        # else:
        #     x_[0] = sliced_noise_feat[beg_i:beg_i + length,:,:,:]
        # print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
        feed = {model.input_x: sliced_noise_feat, model.is_training: False}
        decode = sess.run(model.predictions, feed)
        # if pad > 0:
        #     print('Removing padding of {} samples'.format(pad))
        #     # get rid of last padded samples
        #     canvas_w = canvas_w[:-pad]
        # if c_res is None:
        #     c_res = canvas_w
        # else:
        #     c_res = np.concatenate((c_res, canvas_w))
    return c_res


def main(_):
    depth = 9
    use_he_uniform = True
    optional_shortcut = False
    currentPath = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(currentPath, args.model_path)

    if not os.path.exists(save_path):
        raise ValueError("{} not exists!!".format(save_path))
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = args.model_name

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    cnn_model = VDCNN(
        input_dim=[context_window_size, feat_size],
        batchsize=batchsize,
        depth=9,
        downsampling_type=args.downsampling_type,
        use_he_uniform=use_he_uniform,
        optional_shortcut=optional_shortcut)
    saver = tf.train.Saver()

    saver.restore(sess, os.path.join(save_path, ckpt_name))
    print("[*] Read {}".format(ckpt_name))

    TFRecord = os.path.join(currentPath, os.path.join("data", args.TFRecord))
    num_example = 0
    for record in tf.python_io.tf_record_iterator(TFRecord):
        num_example += 1
    print("total examples in TFRecords {} : {}".format(TFRecord, num_example))

    sliced_feat, sliced_noise_feat_op = read_and_decode(
        TFRecord, context_window_size, feat_size)
    with sess:
        sliced_noise_feat = sess.run(sliced_noise_feat_op)
        clean_feat = clean(sess, cnn_model, sliced_noise_feat)
    print("good done.")

    # with sess:
    #     for i in range(num_iters):
    #         print(i)
    #         sliced_feat, sliced_noise_feat = sess.run(
    #             [sliced_feat_op, sliced_noise_feat_op])
    #         feed = {
    #             cnn_model.input_x: sliced_noise_feat,
    #             cnn_model.input_y: sliced_feat,
    #             cnn_model.is_training: True
    #         }
    #         _, step, loss, accuracy = sess.run(
    #             [train_op, global_step, cnn_model.loss, cnn_model.accuracy],
    #             feed)
    #         train_summary = sess.run(
    #             merge_summary,
    #             feed_dict={
    #                 cnn_model.input_x: sliced_noise_feat,
    #                 cnn_model.input_y: sliced_feat,
    #                 cnn_model.is_training: True
    #             })
    #         print("step {}, Epoch {}, loss {:g}, accuracy {}".format(
    #             step, num_batchs, loss, accuracy))
    #         if i % save_freq == 0 or i == (num_iters - 1):
    #             saver.save(
    #                 sess, os.path.join(saver_path, "saver"), global_step=i)
    #             writer.add_summary(train_summary, step)


if __name__ == "__main__":
    tf.app.run()
