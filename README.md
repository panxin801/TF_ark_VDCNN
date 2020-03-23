# TF_ark_VDCNN

Author: Xin Pan

Date: 2020.03.20

---

This is a self-experiment code repo. Used for speech enhancement  experiment.

Till now, it has two training protocal. One is using ARK files(from kaldi) and the other is using Raw wave files as input. From model archietecture it has two model:

1.  VDCNN-very deep cnn network;
2. ANFCN-fully cnn model designed as auto encoder-decoder.

## Dependiences
This experiment is based on
1. Python3.5.4 64bit
2. Tensorflow-cpy version 1.9.0 AMD64 or
   Tensorflow-gpu version 1.12.0 AMD64
3. Scipy 1.0.0

## How to make TFRecord

This program supports two formation transforming as TFRecord. One is ARK file and the other is raw wave files. We have two Step1, you may choose one from the following two Step1.

### Step1.ARK file

Original **feats.scp** stores the `uttid ark-feats` relationship. For now, I want to process the data as following steps: 
Clean and Noise data have same IDs in parallel, but save in different dirs or files. Both of them have the file named `feats.scp`, I foucs on the `feats.scp` from noise dir and then do some modifies on it.  
Using the following code:

```bash
%s/:.*// # delete the characters after:
and then using vi block delete
```
save as **noise_feats.scp**. After modified, it looks like:
```tex
10000441 1
10000442 1
10000443 1
10000444 1
10000445 1
10000446 1
10000447 1
10000448 1
10000449 1
10000450 1
10000451 1
10000452 1
10000453 1
10000454 1
10000455 1
```
the second block means which noise ark file can find the feature of ID.  

When open a clean ark file read an ID and open `noise_feats.scp` to find counterpart stores in `noise_feats.scp`, then open the noise ark file get the noise feature. Finally, zip the clean feature and the noise feature, store them in TFRecord file.

### Step1.RAW WAVE

In the part, we have two dirs named `clean_dir` and `noise_dir`, and files in both dir have same name is much easier for processing in the next step.

### Step2 Run make_tfrecord.py

This code will make right TFRecord file and save them.

## Train

This part contains two different pipeline either. One is for training ARK file, you can use files named like `*_main` or file don't have suffix. The other is for RAW WAVE formation, it only uses files named like `*_wav`. This is the best difference.

## Note
Data made of 2 parts. One is clean ark file and the other is noise ark file. They must be in parallel relationship, which means one clean file have a counterpart noise file with the same name but in different directory.  
In **wav.scp** they must have the same id, which thet are in different save path and different ark files. The id will used as tag in TFrecord file.

## TODO works
1. how to use tf.Dataset as input pipeline;
2. How to change tf.layers.conv2d change to tf.nn.max_pool;
3. rewrite README;
4. modify the processing routine.