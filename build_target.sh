#!/usr/bin/env bash

BUILD_TYPE="Release"
i=0
args=()
for arg in "$@"
do
    if [ "$arg" = "debug" -o "$arg" = "-DCMAKE_BUILD_TYPE=Debug" ]
    then
        BUILD_TYPE="Debug"
    elif [ "$arg" = "release" -o "$arg" = "-DCMAKE_BUILD_TYPE=Release" ]
    then
        BUILD_TYPE="Release"
    else
        args[$i]="$arg"
        ((++i))
    fi    
done

# Suck in the environment settings for pkg-conf
. /usr/local/sdk-imx6/environment-setup

BUILD_TARGET=armv7l
BUILD_DIR="${BUILD_TARGET}-${BUILD_TYPE}"

if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi

BUILD_DIR_PATH=$(dirname ${BUILD_DIR})

cd ${BUILD_DIR}

if [ ! -e Makefile ]; then
  cmake -DCMAKE_SYSTEM_PROCESSOR=${BUILD_TARGET} \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DIR_PATH}/install" \
        -DCMAKE_TOOLCHAIN_FILE=/usr/local/sdk-imx6/cmake.toolchain \
        "${args[@]}" ..
  if [ $? != 0 ]; then
    exit $_
  fi
fi

make

exit $?
