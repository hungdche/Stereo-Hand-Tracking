#! bin/bash

FILE=libtorch/

if [ ! -d "$FILE" ]; then
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip
else
    echo "$FILE exists"
fi