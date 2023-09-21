#!/usr/bin/env bash
set -e
echo "Installing OpenVINO with version 2023.0.0"
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | tee /etc/apt/sources.list.d/intel-openvino-2023.list
apt update
apt upgrade -y
apt search openvino
apt-get -y install openvino-2023.0.2
export PATH=/usr/lib/x86_64-linux-gnu:$PATH
