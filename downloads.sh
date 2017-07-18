#!/usr/bin/env bash

# Download data
wget https://transfer.d2.mpi-inf.mpg.de/joon/joon17iccv/data.tar.gz
echo "Untar data"
tar xf data.tar.gz
rm data.tar.gz

mkdir cache
mv data/nugu_svm cache/nugu_svm
mv data/nugu_train cache/nugu_train