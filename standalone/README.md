# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a standalone `opt`-like tool to operate on that dialect.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## My building
```sh
mkdir build && cd build

cmake -G Ninja \
-DMLIR_DIR=$PWD/../../installed/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=$PWD/../../build/bin/llvm-lit \
-DCMAKE_INSTALL_PREFIX=$(pwd)/../installed \
../

ninja -j2
ninja check-standalone
```
## run examples
Assume there are some examples in example directory in Linalg Dialect
```sh
./build/bin/standalone-opt ./examples/torch_linear.mlir -linalg-to-standalone -standalone-to-LLVM > ./examples/llvm_linear.mlir
```
It will dump llvm mlir file to ./examples/llvm_linear.mlir. After
```sh
./build/bin/standalone-translate ./examples/llvm_linear.mlir -mlir-to-llvmir > ./examples/llvm_linear.ll
```
It will dump llvm file to ./examples/llvm_linear.ll. If you want to utilize llvm O3 pass
```sh
../installed/bin/opt -S examples/llvm_linear.ll -O3 -o examples/llvm_linear_O3.ll
```
create bitcode file corresponding to the .ll file
```sh
../installed/bin/llvm-as ./examples/llvm_linear_O3.ll
```
execute the file
```sh
./installed/bin/lli ./examples/llvm_linear_O3.bc
```
an example for run pass using pass pipeline
```sh
./build/bin/standalone-opt ./examples/linear_tosa.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg))"
```
