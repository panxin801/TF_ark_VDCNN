import os, sys
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="read and train VDCNN")
parser.add_argument(
    "--TFRecord",
    type=str,
    default="output_TFRecord.tfrecords",
    help="Path and name of TFRecord file.")
args = parser.parse_args()


def read_and_decode(TFRecordqueue, context_window_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(TFRecordqueue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "wav_feat": tf.FixedLenFeature([], tf.string),
            "noisy_feat": tf.FixedLenFeature([], tf.string)
        })
    clean_feats = tf.decode_raw(features["wav_feat"],
                                tf.half)  # tf.half===tf.float16
    noise_feats = tf.decode_raw(features["noisy_feat"], tf.half)
    print(clean_feats.shape)
    clean_feats.set_shape([11, 43])
    print(clean_feats.shape)
    # noise_feats.set_shape(17, 11, 43)
    return clean_feats, noise_feats


def main(_):
    currentPath = os.path.dirname(os.path.abspath(__file__))
    TFRecord = os.path.join(currentPath, os.path.join("data", args.TFRecord))
    TFRecordqueue = tf.train.string_input_producer([TFRecord])
    clean_feats, noise_feats = read_and_decode(
        TFRecordqueue, 11)  # 11 means context window size


if __name__ == "__main__":
    tf.app.run()
    print("Done!")