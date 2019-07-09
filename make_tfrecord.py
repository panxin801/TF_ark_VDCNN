import os, sys
import time
import argparse
import numpy as np
import shutil
import tensorflow as tf


def ReadARKFile(readFile):
    valueList = []
    num = 0
    # ID of what started, and care about the bits
    searchRetId = 0
    for line in open(readFile, "rt"):
        if searchRetId == 0:
            if "{}  [".format(retId) not in line:
                continue
            else:
                searchRetId = 1
        if "[" in line:
            tagid = line.split(" ")[0]
            if (tagid < retId):
                continue
        elif "]" in line:
            Tmpline = line.strip().split("]")[0]
            Tmpline = Tmpline.strip().split(" ")
            valueList.append(Tmpline)
            mel_spectrogram = np.array(valueList)
            mel_spectrogram = mel_spectrogram.T
            warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(
                mel_spectrogram=mel_spectrogram,
                time_warping_para=5,
                time_masking_para=15)

            valueList.clear()
            num += 1
            if (num % 500 == 0):
                print(num)
        else:
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

    # Make noise_feats.scp dict
    noise_feats = os.path.join(data_dir, "noise_feats.scp")
    noise_feats_dict = {}
    with open(noise_feats, "rt", encoding="utf-8") as f:
        for line in f.readlines():
            key, val = line.split()
            noise_feats_dict.update({key: val})

    outTFRecord = "output_TFRecord.tfrecords"
    outputPath = os.path.join(data_dir, outTFRecord)
    if os.path.exists(outputPath):
        os.unlink(outputPath)  # Delete $outputPath this file
        # raise ValueError(
        #     "ERROR: {} already exists. Delete this file or {TODO}.".format(
        #         outputPath))
    # TODO set some cover or judge or delete mechanism with $outputPath( this file )
    output_file = tf.python_io.TFRecordWriter(outputPath)
    for __, clean_ARK_file in enumerate(clean_list):
        ReadARKFile(clean_ARK_file)

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
