set -e
echo "Installing OpenVINO with version ${OPENVINO_VERSION}"
curl -sSL https://apt.repos.intel.com/openvino/$OPENVINO_YEAR/GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR >./GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR
apt-key add ./GPG-PUB-KEY-INTEL-OPENVINO-$OPENVINO_YEAR
echo "deb https://apt.repos.intel.com/openvino/$OPENVINO_YEAR all main" | tee /etc/apt/sources.list.d/intel-openvino-$OPENVINO_YEAR.list
apt update
apt install -y intel-openvino-runtime-ubuntu20-$OPENVINO_VERSION

source /opt/intel/openvino_2021/bin/setupvars.sh
ldconfig

cd ./WasmEdge
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DWASMEDGE_BUILD_TESTS=ON -DWASMEDGE_WASINN_BUILD_OPENVINO=ON .
cmake --build build
