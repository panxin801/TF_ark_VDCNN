import os, sys
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.client import device_lib
from model import VDCNN
devices = device_lib.list_local_devices()

parser = argparse.ArgumentParser(description="read and train VDCNN")
parser.add_argument(
    "--TFRecord",
    type=str,
    default="output_TFRecord.tfrecords",
    help="Path and name of TFRecord file.")
parser.add_argument(
    "--downsampling-type",
    type=str,
    default="maxpool",
    help=
    "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')"
)
args = parser.parse_args()


def read_and_decode(TFRecordqueue, context_window_size, feat_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(TFRecordqueue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "wav_feat": tf.FixedLenFeature([], tf.string),
            "noisy_feat": tf.FixedLenFeature([], tf.string)
        })
    # Check here http://osask.cn/front/ask/view/289151
    clean_feats = tf.decode_raw(features["wav_feat"],
                                tf.float16)  # tf.half===tf.float16
    clean_feats.set_shape([context_window_size * feat_size])
    sliced_feat = tf.reshape(clean_feats, [context_window_size, feat_size])
    noise_feats = tf.decode_raw(features["noisy_feat"], tf.float16)
    noise_feats.set_shape([context_window_size * feat_size])
    sliced_noise_feat = tf.reshape(noise_feats,
                                   [context_window_size, feat_size])
    return sliced_feat, sliced_noise_feat


def main(_):
    # Set some Top params
    batchsize = 32
    context_window_size = 11
    feat_size = 43
    depth = 9
    use_he_uniform = True
    optional_shortcut = False
    currentPath = os.path.dirname(os.path.abspath(__file__))

    TFRecord = os.path.join(currentPath, os.path.join("data", args.TFRecord))
    TFRecordqueue = tf.train.string_input_producer([TFRecord])
    clean_feats, noise_feats = read_and_decode(
        TFRecordqueue, context_window_size,
        feat_size)  # 11 means context window size
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    udevice = []
    for device in devices:
        if len(devices) > 1 and "CPU" in device:
            continue
        udevice.append(device)
    sess = tf.Session(config=config)
    cnn_model = VDCNN(
        input_dim=[context_window_size, feat_size],
        batchsize=batchsize,
        depth=9,
        downsampling_type=args.downsampling_type,
        use_he_uniform=use_he_uniform,
        optional_shortcut=optional_shortcut)


if __name__ == "__main__":
    # try:
    #     tf.app.run()
    # except Exception as e:
    #     print("Error: ", e)
    # finally:
    #     print("Done!")
    tf.app.run()
