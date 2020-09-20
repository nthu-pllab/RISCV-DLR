# RISC-V Custom Runtime (DLR)
This repository is a lightweight runtime for RISC-V utility. We uses [TVM](https://github.com/apache/incubator-tvm) runtime to create a C++ interface that can be called by `host.cpp` to invoke runtime function. After saving a TVM module as `.ll`, `.graph` and `.params`, with supporting from LLVM and `host.cpp`. Users can further compile the program with riscv-gun-toolcahin and easily run the inference with Spike.

## How to use
![](https://i.imgur.com/HJ4nOv2.png)

## Build steps

There are 2 targets in DLR.
- A static library: `libDLR.a`
    - for compiling with host code
- An executable: `DumpKernel`
    - for generating kernel.inc from graph

Because `libDLR.a` is for compiling with host code and eventually be executed on Spike, it need to be compiled by gcc/g++ of riscv-gnu-toolchain.

On the other hand, `DumpKernel` can be executed directly, so it has no need to be compiled by gcc/g++ of riscv-gnu-toolchain.

```sh
##############################
#  RISCV-build for libDLR.a  #
##############################
# set RISCV to where riscv-gnu-toolchain installed
mkdir build-riscv
cd build-riscv

CC=$RISCV/bin/riscv64-unknown-elf-gcc \
  CXX=$RISCV/bin/riscv64-unknown-elf-g++ \
  cmake .. \
  -DCMAKE_INSTALL_PREFIX=../install-riscv
cmake --build .
cmake --install .

cd ..

#################################
#  normal build for DumpKernel  #
#################################
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=../install
cmake --build .
cmake --install .

cd ..
```

## Example - Pre-quantized mobilenet v1 model
The example is shown in `example/pre_quant_mobilenet_v1_tflite`
- In `build_model.py`, we run the TVM routine and save the TVM module.
- In `host.cpp`, we invoke function in DLR for init, load input, and running. 

After setting up the paths of dependences in Makefile, you can run all the flow easily, as illustrated in the figure above.

- **RISCV_INSTALL**: where riscv-gnu-toolchain is installed
- **DLR_INSTALL**: as the explanation in "Build Steps" section, we need to build DLR twice for different purpose. This one is for `DumpKernel` executable to generate `kernel.inc`.
- **DLR_RISCV_INSTALL**: This one is for `libDLR.a` static library, which is compiled by gcc/g++ of riscv-gnu-toolchain. The static library will be compiled with the host code and run on Spike.
- **LLVM_INSTALL**: modified LLVM with RISC-V P extension support.

```shell
make
make run
```

## Dependent project
- LLVM
    - The version we used as target for TMV QNN with RISC-V P extension support is open source at [here](https://github.com/nthu-pllab/llvm9-project-rvp).
- riscv-gun-toolchain
    - To support our flow for TVM QNN, please use [riscv-binutils](https://github.com/nthu-pllab/riscv-binutils-rvp) and [spike](https://github.com/nthu-pllab/riscv-isa-sim) that we modified for supporting P extension

### Acknowledgement
- Apache TVM

