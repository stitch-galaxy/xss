#download and install CUDA 8

#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
fi

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

vi ~/.profile

# check that cuda and NVidia driver installed
nvidia-smi

# download and install cudnn 5.1
# register on NVidia site
# download deb packages
# move to tgt machine
# sudo dpkg -i [name].deb

#  install libcupti-dev
sudo apt-get install libcupti-dev

#download anaconda: https://www.continuum.io/downloads
curl -O https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
sudo bash Anaconda3-4.4.0-Linux-x86_64.sh
conda create -n sg python=3.5
source activate sg
pip install tensorflow-gpu


# check tensorflow see GPU
>>>import tensorflow as tf
>>>sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))