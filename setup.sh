#!/bin/bash
# sudo apt install npm
# sudo npm install --global xpm
# xpm install --global @xpack-dev-tools/riscv-none-embed-gcc@latest
git submodule update --init
if [[ -z "${NO_BUILD}" ]]; then
	pushd emulator
	mkdir -p build
	pushd build
	cmake .. && make -j4
	popd
fi
