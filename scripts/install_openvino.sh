#!/usr/bin/env bash

if [[ ! -v "${OPENVINO_VERSION}" ]]; then
  OPENVINO_VERSION="2021.4.582"
fi
if [[ ! -v "${OPENVINO_YEAR}" ]]; then
  OPENVINO_YEAR="2021"
fi

set -e
echo "Installing OpenVINO with version ${OPENVINO_VERSION}"
curl -sSL https://apt.repos.intel.com/openvino/$OPENVINO_YEAR/GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR | gpg --dearmor > /usr/share/keyrings/GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR.gpg
echo "deb [signed-by=/usr/share/keyrings/GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR.gpg] https://apt.repos.intel.com/openvino/$OPENVINO_YEAR all main" | tee /etc/apt/sources.list.d/intel-openvino-$OPENVINO_YEAR.list
apt update
apt install -y intel-openvino-runtime-ubuntu20-$OPENVINO_VERSION
source /opt/intel/openvino_2021/bin/setupvars.sh
ldconfig

echo "*** inspect /etc/ld.so.conf"
cat /etc/ld.so.conf

echo "*** add new paths to /etc/ld.so.conf"
echo "include /etc/ld.so.conf.d/*.conf" >> /etc/ld.so.conf
echo "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/" >> /etc/ld.so.conf
echo "/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/" >> /etc/ld.so.conf

echo "*** inspect /etc/ld.so.conf"
cat /etc/ld.so.conf