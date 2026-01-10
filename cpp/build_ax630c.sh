mkdir -p build_ax630c && cd build_ax630c
cmake ..  \
  -DCHIP_AX630C=ON  \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX=../install/ax630c \
  -DCMAKE_BUILD_TYPE=Release
make -j4
make install