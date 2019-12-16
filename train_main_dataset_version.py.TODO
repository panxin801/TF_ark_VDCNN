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


def _parse_record(example_proto):
    features = {
        "wav_feat": tf.FixedLenFeature([], tf.string),
        "noisy_feat": tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features=features)
    parsed_features["wav_feat"] = tf.decode_raw(parsed_features["wav_feat"],
                                                tf.half)
    parsed_features["noisy_feat"] = tf.decode_raw(
        parsed_features["noisy_feat"], tf.half)
    return parsed_features


def read_and_decode(TFRecordqueue, context_window_size, feat_size):
    dataset = tf.data.TFRecordDataset(TFRecordqueue)
    dataset = dataset.map(_parse_record)
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        # Check here http://osask.cn/front/ask/view/289151
        clean_feats.set_shape([context_window_size * feat_size])
        sliced_feat = tf.reshape(clean_feats, [context_window_size, feat_size])
        noise_feats.set_shape([context_window_size * feat_size])
        sliced_noise_feat = tf.reshape(noise_feats,
                                       [context_window_size, feat_size])

    return sliced_feat, sliced_noise_feat


def main(_):
    try:
        # Set some Top params
        batchsize = 128

        currentPath = os.path.dirname(os.path.abspath(__file__))
        TFRecord = os.path.join(currentPath, os.path.join(
            "data", args.TFRecord))
        clean_feats, noise_feats = read_and_decode(
            TFRecord, 11, 43)  # 11 means context window size
        # cleanbatch, noisebatch = tf.train.shuffle_batch(
        #     [clean_feats, noise_feats],
        #     batchsize,
        #     num_threads=2,
        #     capacity=1000 + 3 * batchsize,
        #     min_after_dequeue=1000,
        #     name="clean_and_noise")
    except Exception as e:
        print("Error: ", e)
    finally:
        print("All good!")


if __name__ == "__main__":
    tf.app.run()
    print("Done!")