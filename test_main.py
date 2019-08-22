import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from model import VDCNN
import numpy as np

parser = argparse.ArgumentParser(
    description="Test VDCNN model using test ARK files.")
parser.add_argument(
    "--model-path", type=str, default="model_save", help="Model save dir.")
parser.add_argument(
    "--model-name", type=str, default="saver-6", help="Saver name")
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


def ReadARKFile(sess, readFiles, cnn_model):
    valueList = []
    num = 0
    findHead = 0

    for readFile in readFiles:
        # ID of what started, and care about the bits
        for line in open(readFile, "rt"):
            if "  [" in line and findHead == 0:
                tagid = line.split(" ")[0]
                findHead = 1
            elif "]" in line and findHead == 1:
                Tmpline = line.strip().split("]")[0]
                Tmpline = Tmpline.strip().split(" ")
                valueList.append(Tmpline)
                feat_spectrogram = np.array(valueList, dtype=np.float32)
                encoder_proc(sess, feat_spectrogram, cnn_model)
                valueList.clear()
                num += 1
                if (num % 500 == 0):
                    print(num)
                findHead = 0
            elif findHead == 1:
                Tmpline = line.strip()
                valueList.append(Tmpline.split(" "))


def encoder_proc(sess, feat_spectrogram, cnn_model):
    sliced_feat_spectrogram = slice_wav(feat_spectrogram)
    if sliced_feat_spectrogram.shape[0] > batchsize:
        for begin_i in range(batchsize, sliced_feat_spectrogram.shape[0],
                             batchsize):
            batch_feat = sliced_feat_spectrogram[begin_i - batchsize:
                                                 begin_i, :, :]
            # batch_feat = tf.reshape(
            #     batch_feat, [batchsize, context_window_size, feat_size, 1])
            batch_feat = batch_feat.reshape(
                [batchsize, context_window_size, feat_size, 1])
            with sess:
                clean(sess, cnn_model, batch_feat)


# and then flatten the input. But if the shape of wav signals less then (time,freq), then padding with 0
def slice_wav(input_signals, windows_size=11, freq_size=43, stride=1):
    assert input_signals.shape[1] == freq_size
    n_samples = input_signals.shape[0]
    offset = int(stride * windows_size)
    retSlice = []
    for start in range(0, n_samples, offset):
        # Less than 1 signal block
        end = start + offset
        if end > n_samples:
            end = n_samples
        oneSlice = input_signals[start:end]
        if oneSlice.shape[0] == windows_size:
            retSlice.append(oneSlice)
        elif oneSlice.shape[0] < windows_size:
            tmpFillZero = np.zeros(
                [windows_size - oneSlice.shape[0], freq_size], np.float32)
            oneSlice = np.concatenate((oneSlice, tmpFillZero))
            retSlice.append(oneSlice)
    return np.array(retSlice, np.float32)


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

    test_path = os.path.join(currentPath, os.path.join("data", "test"))
    test_list = [
        os.path.join(test_path, file) for file in os.listdir(test_path)
        if file.endswith(".txt")
    ]
    ReadARKFile(sess, test_list, cnn_model)

    # sliced_feat, sliced_noise_feat_op = read_and_decode(
    #     TFRecord, context_window_size, feat_size)
    # with sess:
    #     sliced_noise_feat = sess.run(sliced_noise_feat_op)
    #     clean_feat = clean(sess, cnn_model, sliced_noise_feat)
    # print("good done.")


if __name__ == "__main__":
    tf.app.run()
