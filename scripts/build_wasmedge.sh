set -e

source /opt/intel/openvino_2021/bin/setupvars.sh
ldconfig

cd ./WasmEdge
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_WASINN_BUILD_OPENVINO=ON .
cmake --build build
