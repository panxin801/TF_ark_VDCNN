import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.client import device_lib
from model import VDCNN, ANFCN
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
# Set some global variables
context_window_size = 11
feat_size = 43
batchsize = 256  # 512->10
num_epochs = 4
save_freq = 100
model_type = "ANFCN"


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
    dataset = dataset.map(_parse_record).repeat().batch(batchsize).shuffle(
        batchsize * 10, seed=32)
    data_iterator = dataset.make_one_shot_iterator()
    sliced_feat, sliced_noise_feat = data_iterator.get_next()
    return sliced_feat, sliced_noise_feat


def main(_):
    # Set some Top params
    # batchsize = 32
    # context_window_size = 11
    # feat_size = 43
    # Change the 2 params to global params
    depth = 9
    use_he_uniform = True
    optional_shortcut = False
    learning_rate = 1e-3
    # num_epochs = 3
    currentPath = os.path.dirname(os.path.abspath(__file__))

    saver_path = os.path.join(currentPath, "model_save")
    if not os.path.exists(saver_path):
        os.mkdir(saver_path)  # create save dir

    TFRecord = os.path.join(currentPath, os.path.join("data", args.TFRecord))
    num_example = 0
    for record in tf.python_io.tf_record_iterator(TFRecord):
        num_example += 1
        # if num_example == 1:  # new add!!!!
        #     break
    print("#############################total examples in TFRecords {} : {}".
          format(TFRecord, num_example))
    num_batchs = num_example / batchsize
    num_iters = int(num_batchs * num_epochs) + 1

    sliced_feat_op, sliced_noise_feat_op = read_and_decode(
        TFRecord, context_window_size, feat_size)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    udevice = []
    for device in devices:
        if len(devices) > 1 and device.device_type == "GPU":
            continue
        udevice.append(device)
    sess = tf.Session(config=config)
    if model_type == "ANFCN":
        cnn_model = ANFCN(
            input_dim=[context_window_size, feat_size],
            batchsize=batchsize,
            is_ref=True,
            do_prelu=True)
    elif model_type == "VDCNN":
        cnn_model = VDCNN(
            input_dim=[context_window_size, feat_size],
            batchsize=batchsize,
            depth=9,
            downsampling_type=args.downsampling_type,
            use_he_uniform=use_he_uniform,
            optional_shortcut=optional_shortcut)
    else:
        print("Model type error!!")
        sys.exit(1)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("2!!!!!!")
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # TODO: change the num_batches_per_epoch update strategynum_epochs*num_batches_per_epoch
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, num_epochs, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        gradients, variables = zip(
            *optimizer.compute_gradients(cnn_model.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables), global_step=global_step)
    print("Initializing all variables.")
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(os.path.join(saver_path, "train")):
        os.mkdir(os.path.join(saver_path, "train"))
    tf.summary.scalar("loss", cnn_model.loss)
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(
        os.path.join(saver_path, "train"), sess.graph)
    saver = tf.train.Saver()  # local model saver

    num_iters = 101
    with sess:
        for i in range(num_iters):
            sliced_feat, sliced_noise_feat = sess.run(
                [sliced_feat_op, sliced_noise_feat_op])
            feed = {
                cnn_model.input_x: sliced_noise_feat,
                cnn_model.input_y: sliced_feat,
                cnn_model.is_training: True
            }
            _, step, loss = sess.run([train_op, global_step, cnn_model.loss],
                                     feed)
            train_summary = sess.run(
                merge_summary,
                feed_dict={
                    cnn_model.input_x: sliced_noise_feat,
                    cnn_model.input_y: sliced_feat,
                    cnn_model.is_training: True
                })
            print("step {}/{}, loss {:g}".format(step, num_iters, loss))
            if i % save_freq == 0 or i == (num_iters - 1):
                saver.save(
                    sess, os.path.join(saver_path, "saver"), global_step=i)
                writer.add_summary(train_summary, step)
                # try:
                #     pass
                # except tf.errors.OutOfRangeError:
                #     print("Done training, epoch limit {} reached.".format(num_epochs))
                # finally:


if __name__ == "__main__":
    # try:
    #     tf.app.run()
    # except Exception as e:
    #     print("Error: ", e)
    # finally:
    #     print("Done!")
    tf.app.run()