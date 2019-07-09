# TF_ark_VDCNN
Using tensorflow and ARK files for VDCNN 

## Dependicences
This experiment is based on
1. Python3.5.4 64bit
2. Tensorflow-cpy version 1.9.0 AMD64

## Note
Data made of 2 parts. One is clean ark file and the other is noise ark file. They must be in parallel relationship, which means one clean file have a counterpart noise file with the same name but in different directory.  
In **wav.scp** they must have the same id, which thet are in different save path and different ark files. The id will used as tag in TFrecord file.