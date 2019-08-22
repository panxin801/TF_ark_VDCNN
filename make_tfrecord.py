import os, sys
import argparse
import numpy as np
import shutil
import tensorflow as tf


def findNoiseARK(tagid, noise_ARKID, noise_list):
    valueList = []
    findHead = 0

    for name in noise_list:
        noise_list_id = name.split(".")[-2]
        if noise_ARKID != noise_list_id:
            continue
        for line in open(name, "rt"):
            if "{}  [".format(tagid) in line and findHead == 0:
                findHead = 1
            elif "]" in line and findHead == 1:
                Tmpline = line.strip().split("]")[0]
                Tmpline = Tmpline.strip().split(" ")
                valueList.append(Tmpline)
                feat_spectrogram = np.array(valueList, dtype=np.float32)
                valueList.clear()
                findHead = 0
                return feat_spectrogram
            elif findHead == 1:
                Tmpline = line.strip()
                valueList.append(Tmpline.split(" "))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# FOr slice original wav signals into fixed block like (time,freq)=(11,43),
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


def encoder_proc(feat_spectrogram, noise_feat_spectrogram, output_file):
    sliced_feat_spectrogram = slice_wav(feat_spectrogram)
    sliced_noise_feat_spectrogram = slice_wav(noise_feat_spectrogram)
    assert sliced_feat_spectrogram.shape == sliced_noise_feat_spectrogram.shape, sliced_feat_spectrogram.shape
    for (feat, noise_feat) in zip(sliced_feat_spectrogram,
                                  sliced_noise_feat_spectrogram):
        feat = feat.flatten()
        noise_feat = noise_feat.flatten()
        feat_str = feat.tostring()
        noise_feat_str = noise_feat.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'wav_feat': _bytes_feature(feat_str),
                    'noisy_feat': _bytes_feature(noise_feat_str)
                }))
        output_file.write(example.SerializeToString())

    ## sliced_feat_spectrogram, sliced_noise_feat_spectrogram are sliced feats.
    ## They are now feats block already with dims[window_num, time,freq]
    # feat_str = sliced_feat_spectrogram.tostring()
    # noise_feat_str = sliced_noise_feat_spectrogram.tostring()
    # example = tf.train.Example(
    #     features=tf.train.Features(
    #         feature={
    #             'wav_feat': _bytes_feature(feat_str),
    #             'noisy_feat': _bytes_feature(noise_feat_str)
    #         }))
    # output_file.write(example.SerializeToString())


def ReadARKFile(readFile, noise_list, output_file):
    valueList = []
    noise_feat_spectrogram = []
    num = 0
    findHead = 0
    noise_ARKID = 0

    # ID of what started, and care about the bits
    for line in open(readFile, "rt"):
        if "  [" in line and findHead == 0:
            tagid = line.split(" ")[0]
            noise_ARKID = readFile.split(".")[1]
            noise_feat_spectrogram = findNoiseARK(tagid, noise_ARKID,noise_list)
            findHead = 1
        elif "]" in line and findHead == 1:
            Tmpline = line.strip().split("]")[0]
            Tmpline = Tmpline.strip().split(" ")
            valueList.append(Tmpline)
            feat_spectrogram = np.array(valueList, dtype=np.float32)
            encoder_proc(feat_spectrogram, noise_feat_spectrogram, output_file)
            valueList.clear()
            num += 1
            if (num % 500 == 0):
                print(num)
            findHead = 0
        elif findHead == 1:
            Tmpline = line.strip()
            valueList.append(Tmpline.split(" "))


def main():
    currentDir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(currentDir, "data")
    clean_dir = os.path.join(data_dir, "clean")
    noise_dir = os.path.join(data_dir, "noise")

    clean_list = [
        os.path.join(clean_dir, file) for file in os.listdir(clean_dir)
        if file.endswith(".txt")
    ]
    noise_list = [
        os.path.join(noise_dir, file) for file in os.listdir(noise_dir)
        if file.endswith(".txt")
    ]

    outTFRecord = "output_TFRecord.tfrecords"
    outputPath = os.path.join(data_dir, outTFRecord)
    if os.path.exists(outputPath):
        os.unlink(outputPath)  # Delete $outputPath this file
        raise ValueError(
            "ERROR: {} already exists. Delete this file or {TODO}.".format(
                outputPath))
    #TODO set some cover or judge or delete mechanism with $outputPath( this file )
    output_file = tf.python_io.TFRecordWriter(outputPath)
    for __, clean_ARK_file in enumerate(clean_list):
        ReadARKFile(clean_ARK_file, noise_list, output_file)

    output_file.close()
    print("Done!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert txt featyre to TFRecord.')
    parser.add_argument(
        '--audio-path', default='/home/panxin', help='The audio file.')
    parser.add_argument(
        '--force-gen',
        dest='force_gen',
        action='store_true',
        help='Flag to force overwriting existing dataset.')
    parser.set_defaults(force_gen=True)
    args = parser.parse_args()
    main()
