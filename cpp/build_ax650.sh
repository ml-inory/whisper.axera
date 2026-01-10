mkdir -p build_ax650 && cd build_ax650
cmake ..  \
  -DCHIP_AX650=ON  \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/ax650 \
  -DCMAKE_BUILD_TYPE=Release
make -j4
make install