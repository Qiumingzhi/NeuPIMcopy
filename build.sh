rm -rf build
mkdir build && cd build
conan install .. --build missing --output-folder=. -o *:fPIC=True
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
make -j30 2>&1 | tee ../build.log