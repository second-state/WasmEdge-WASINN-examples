#!/usr/bin/env bash
set -e
echo "Installing OpenVINO with version 2024.2.0"
KEY_FILE=GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
wget https://apt.repos.intel.com/intel-gpg-keys/$KEY_FILE && \
    apt-key add $KEY_FILE && \
    rm -f $KEY_FILE
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu24 main" | tee /etc/apt/sources.list.d/intel-openvino-2024.list
apt update
apt upgrade -y
apt-get -y install openvino-2024.2.0
export PATH=/usr/lib/x86_64-linux-gnu:$PATH
ldconfig
