#!/bin/bash

# Create the data directory in the parent directory of the scripts directory
mkdir -p ../data

# Download and extract the VoxCeleb1 dev data
if [ ! -d "../data/VoxCeleb1/dev" ]; then
  mkdir -p ../data/VoxCeleb1
  download_dir="../data/VoxCeleb1"
  for part in a b c d; do
    wget --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta${part} -P ${download_dir} &
  done
  wait
  cat ${download_dir}/vox1_dev* >${download_dir}/vox1_dev_wav.zip
  unzip ${download_dir}/vox1_dev_wav.zip -d ${download_dir}/dev
fi

# Download and extract the VoxCeleb1 test data
if [ ! -d "../data/VoxCeleb1/test" ]; then
  mkdir -p ../data/VoxCeleb1
  wget --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip -P ../data/VoxCeleb1
  unzip ../data/VoxCeleb1/vox1_test_wav.zip -d ../data/VoxCeleb1/test
fi

# move test to dev for the subsequent creation of vox csv files
mv ../data/VoxCeleb1/test/wav/* ../data/VoxCeleb1/dev/wav/
rm -rf ../data/VoxCeleb1/test

wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -P ../data/VoxCeleb1