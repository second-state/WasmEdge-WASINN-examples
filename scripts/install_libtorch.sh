#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: 2019-2022 Second State INC

if [[ ! -n ${PYTORCH_VERSION} ]]; then
  PYTORCH_VERSION="1.8.2"
fi

DOWNLOAD_TO=$1
if [ ! -d $1/libtorch ]; then
    apt update
    apt upgrade -y
    apt-get install unzip -y
    curl -s -L -O --remote-name-all https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip
    echo "b76d6dd4380e2233ce6f7654e672e13aae7c871231d223a4267ef018dcbfb616 libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip" | sha256sum -c
    unzip -q "libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
    rm -f "libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
fi
