import os, sys
import time
import argparse
import numpy as np
import shutil
import tensorflow as tf

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument(
    '--audio-path', default='/home/panxin', help='The audio file.')

args = parser.parse_args()


def writeTXT(writer, tagid, warped_masked_spectrogram):
    print("%s  [" % (tagid), file=writer, flush=True)
    for y in range(0, warped_masked_spectrogram.shape[1]):
        writer.write("  ")
        for x in range(0, warped_masked_spectrogram.shape[0]):
            writer.write("%f " % (warped_masked_spectrogram[x][y]))
        if (y == warped_masked_spectrogram.shape[1] - 1):
            print("]", file=writer, flush=True)
        else:
            writer.write("\n")


# The following new adding func
def checkSaveFile(writeFile):
    if (os.path.exists(writeFile)):
        writeNew = writeFile + "new"
        writer = open(writeNew, "wt")
        for line in open(writeFile):
            if "[" in line:
                retId = line.split("  ")[0]
        for line in open(writeFile):
            if "{}  [".format(retId) in line:
                break
            else:
                print(line, end="", file=writer)
        writer.close()
        shutil.move(writeNew, writeFile)
        return retId


def wrif(writerList, readList, i):
    print("{} read {}, write {}".format(i, readList[i], writerList[i]))
    # Next line can be comment, when dealing with normal ending
    retId = checkSaveFile(writerList[i])
    valueList = []
    num = 0
    writer = open(writerList[i], "at")
    reachRetId = 0
    for line in open(readList[i]):
        if reachRetId == 0:
            if "{}  [".format(retId) not in line:
                continue
            else:
                reachRetId = 1
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
            writeTXT(writer, tagid, warped_masked_spectrogram)
            valueList.clear()
            num += 1
            if (num % 100 == 0):
                print(num)
        else:
            Tmpline = line.strip()
            valueList.append(Tmpline.split(" "))
    writer.close()


def main():
    currentdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(currentdir, "data")
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

    outTFrecord = "output_TFRecord.tfrecords"
    outputPath = os.path.join(data_dir, outTFrecord)
    if os.path.exists(outputPath):
        raise ValueError(
            "ERROR: {} already exists. Delete this file or {TODO}.".format(
                outputPath))
    # TODO set some cover or judge or delete mechanism with $outputPath( this file )


    print("Done!!")


if __name__ == "__main__":
    main()
